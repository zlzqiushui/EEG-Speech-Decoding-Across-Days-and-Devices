"""
Module B v3 — 公共通道版本

与 v2 的差异:
  - 使用 finetune/data_all_v2.py (公共通道版本)。
  - in_dim 固定为 57 (NUM_COMMON_CHANNELS), 不随 eeg_device 变化。
  - 结果保存到 finetune_result_v3。
  - 预训练模型仍从 result_v3 加载。
  - 其余逻辑 (5种策略, transfer 场景, 特征加载等) 与 v2 一致。

用法:
  python finetune/train_finetune_v3.py --subject_id 1 --transfer day1_to_day2 --strategy 2 --data_ratio 1.0 --device cuda:0 --feature_name mel
"""

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
import copy
import os
import os.path as op
import json
import sys
import random

sys.path.insert(0, op.dirname(op.dirname(op.realpath(__file__))))
sys.path.insert(0, op.dirname(op.realpath(__file__)))

from finetune.config_v2 import project_dir
from finetune.model_v1 import BrainNetworkCL
from finetune.data_all_v2 import (KUL_dataset, get_stimulus_feat, get_EEG_emb_feat_story_range,
                                   NUM_COMMON_CHANNELS, COMMON_CHANNELS)

# v3: 预训练模型从 result_v4 加载 (train_models_sub_v4.py 的输出, in_dim=57)
result_v4_dir = op.join(project_dir, 'result_v4')


# =========================================================================
# 默认配置 (对齐 train_models_sub_v4.py)
# =========================================================================
DEFAULTS = dict(
    kernel_size=3,
    att_out_dim=256,
    dropout=0.5,
    model_name='BrainNetworkCL',
    opt='Adam',
    lr_dic={'Adam': 2e-4, 'SGD': 1e-4},
    num_epochs=150,
    early_stop_num=20,
    batch_size=32,
    negsample_num=32,
    device='cuda:0',
    use_multi_band=False,
    seg_len=5,
    fs=250,
    feature_name='envelope',
    feature_dim_dict={'wav2vecbase9': 1024, 'bert': 768, 'envelope': 1, 'mel': 80, 'mel_10': 10},
    valid=1,
    seed=42,
    num_workers=16,
)


# =========================================================================
# 随机种子工具
# =========================================================================
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# =========================================================================
# 迁移场景定义
# =========================================================================
TRANSFER_SCENARIOS = {
    'day1_to_day2': {
        'name': 'Day1 → Day2 (Cross-day)',
        'source': {
            'train': list(range(1, 13)),
            'valid': list(range(13, 16)),
            'test':  list(range(16, 18)),
            'device': 'day1',
        },
        'target': {
            'train': list(range(18, 30)),
            'valid': list(range(30, 32)),
            'test':  list(range(32, 34)),
            'device': 'day2',
        },
        'source_suffix': 'day1',
        'target_suffix': 'day2',
    },
    'day2_to_day1': {
        'name': 'Day2 → Day1 (Cross-day)',
        'source': {
            'train': list(range(18, 30)),
            'valid': list(range(30, 32)),
            'test':  list(range(32, 34)),
            'device': 'day2',
        },
        'target': {
            'train': list(range(1, 13)),
            'valid': list(range(13, 16)),
            'test':  list(range(16, 18)),
            'device': 'day1',
        },
        'source_suffix': 'day2',
        'target_suffix': 'day1',
    },
    'neurascan_to_brk': {
        'name': 'Neurascan → Neuracle (Cross-device)',
        'source': {
            'train': list(range(1, 16)),
            'valid': list(range(16, 18)),
            'test':  list(range(18, 34)),
            'device': 'neurascan',
        },
        'target': {
            'train': list(range(34, 46)),
            'valid': list(range(46, 49)),
            'test':  list(range(49, 51)),
            'device': 'brk',
        },
        'source_suffix': 'neurascan',
        'target_suffix': 'brk',
    },
}


# =========================================================================
# 模型评估/训练函数 (与 v2 相同)
# =========================================================================
def valid_model(model, dataloaders, device, optimizer, embeds, phase='valid'):
    model.eval()
    epoch_induce = None
    sub_ids = []

    for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(tqdm(dataloaders[phase], desc=f'Validation ({phase})'), start=1):
        embed = embeds[emb_id]
        eeg, embed, sub_id = eeg.to(device), embed.to(device), sub_id.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            pred_induce = model.test(x=eeg, sub_id=sub_id, frame_id=emb_id, embed=embed, sti_id=sti_id)

        if epoch_induce is None:
            epoch_induce = pred_induce
        else:
            epoch_induce = torch.cat((epoch_induce, pred_induce))
        sub_ids.append(sub_id)

    sub_ids = torch.cat(sub_ids)
    sub_ids_set = torch.unique(sub_ids).detach().cpu().numpy().tolist()
    sub_names = dataloaders[phase].dataset.sub_names
    acc = epoch_induce == 0
    acc_sub_dict = {}
    for sid in sub_ids_set:
        sub_name = sub_names[sid]
        acc_sub = acc[sub_ids == sid]
        acc_sub_dict[sub_name] = acc_sub.sum().item() / len(acc_sub)

    acc_sub_mean = np.mean(list(acc_sub_dict.values()))
    return acc_sub_mean, acc_sub_dict


def train_model(dataloaders, dataset_sizes, device, model,
                optimizer, scheduler, start_epoch, num_epochs,
                best_top1, negsample_num, embeds,
                checkpoint_path_best,
                early_stop_num,
                logger,
                writer,
                seed=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    if best_top1 is None:
        best_top1 = 0
        best_epoch = -1
    else:
        best_epoch = start_epoch - 1

    global_step = 0
    iter50_loss = 0.0
    iter50_rank = None
    phase = 'train'
    un_update_step = 0

    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        epoch_rank = None

        for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(
                tqdm(dataloaders[phase], desc=f'Train epoch {epoch}'), start=1):
            embed = embeds[emb_id]
            eeg, embed, sub_id = eeg.to(device), embed.to(device), sub_id.to(device)

            optimizer.zero_grad()
            pred, loss, rank = model(x=eeg, sub_id=sub_id, frame_id=emb_id, embed=embed, sti_id=sti_id)
            loss = loss.mean(0, keepdim=True)

            if iter50_rank is None:
                iter50_rank = rank
            else:
                iter50_rank = torch.cat((iter50_rank, rank))
            if epoch_rank is None:
                epoch_rank = rank
            else:
                epoch_rank = torch.cat((epoch_rank, rank))

            iter50_loss += loss.item()
            running_loss += loss.item() * eeg.size()[0]
            loss.backward()
            optimizer.step()
            global_step += 1
            it_num = 100

            if global_step % it_num == 0:
                top10_iter50 = torch.nonzero(iter50_rank <= 10).shape[0] / iter50_rank.shape[0]
                top5_iter50 = torch.nonzero(iter50_rank <= 5).shape[0] / iter50_rank.shape[0]
                top1_iter50 = torch.nonzero(iter50_rank == 1).shape[0] / iter50_rank.shape[0]
                sample_size = negsample_num + 1
                rank_acc_iter50 = (sample_size - torch.mean(iter50_rank.float())) / (sample_size - 1)

                writer.add_scalar('Iter50/Loss', iter50_loss / it_num, global_step // it_num)
                writer.add_scalars('Iter50/Accuracy',
                                   {'top-10': top10_iter50, 'top-5': top5_iter50,
                                    'top-1': top1_iter50, 'rank-acc': rank_acc_iter50},
                                   global_step // it_num)

                logger.info(f"Train: epoch {epoch}/ global_step {global_step}: "
                            f"loss: {iter50_loss / it_num:.4f}; top-1: {top1_iter50:.4f}; "
                            f"rank-acc: {rank_acc_iter50:.4f}")

                print('evaluating valid set...')
                top1_iter50_valid, acc_sub_dict = \
                    valid_model(model, dataloaders, device, optimizer, embeds, phase='valid')
                model.train()

                writer.add_scalars('Iter50/valid-Accuracy', {'top-1': top1_iter50_valid}, global_step // it_num)
                writer.add_scalars('Iter50/valid-Accuracy-sub', acc_sub_dict, global_step // it_num)

                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: top-1: {top1_iter50_valid:.4f}")
                acc_sub_dict_as_str = ', '.join([f'{k}: {v:.4f}' for k, v in acc_sub_dict.items()])
                logger.info(f"Dev: acc_sub_dict: {acc_sub_dict_as_str}")

                if top1_iter50_valid > best_top1:
                    best_top1 = top1_iter50_valid
                    best_epoch = epoch
                    un_update_step = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "epoch": best_epoch,
                        "top1_acc": best_top1,
                        "global_step": global_step,
                        "seed": seed}
                    torch.save(checkpoint, checkpoint_path_best)
                    logger.info(f'update best on valid checkpoint: {checkpoint_path_best}')
                else:
                    un_update_step += 1

                logger.info("\n\n")

        scheduler.step()
        epoch_loss = running_loss / dataset_sizes[phase]

        top10_epoch = torch.nonzero(epoch_rank <= 10).shape[0] / epoch_rank.shape[0]
        top5_epoch = torch.nonzero(epoch_rank <= 5).shape[0] / epoch_rank.shape[0]
        top1_epoch = torch.nonzero(epoch_rank == 1).shape[0] / epoch_rank.shape[0]
        sample_size = negsample_num + 1
        rank_acc_epoch = (sample_size - torch.mean(epoch_rank.float())) / (sample_size - 1)

        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_scalars(f'Accuracy/{phase}',
                           {'top-10': top10_epoch, 'top-5': top5_epoch, 'top-1': top1_epoch,
                            'rank-acc': rank_acc_epoch},
                           epoch)
        logger.info(f"epoch {epoch}: loss/{phase}: {epoch_loss:.5f}; "
                    f"top-10: {top10_epoch:.4f}; top-5: {top5_epoch:.4f}; "
                    f"top-1: {top1_epoch:.4f}; rank-acc: {rank_acc_epoch:.4f}")

        if un_update_step > early_stop_num:
            logger.info(f'early stop at epoch {epoch}, global_step {global_step}')
            print(f'early stop at epoch {epoch}, global_step {global_step}')
            break

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val acc epoch {best_epoch}: {best_top1:.4f}')

    model.load_state_dict(best_model_wts)

    final_checkpoint = {
        "model_state_dict": best_model_wts,
        "optimizer_state_dic": optimizer.state_dict(),
        "epoch": best_epoch if best_epoch >= 0 else 0,
        "top1_acc": best_top1,
        "global_step": global_step,
        "seed": seed}
    torch.save(final_checkpoint, checkpoint_path_best)
    logger.info(f'Final checkpoint saved to {checkpoint_path_best}')

    return model, best_top1, best_epoch


# =========================================================================
# 辅助函数
# =========================================================================
def build_model_tag(args, subject_id, ratio, suffix=''):
    tag_nn = f'-nn{args.negsample_num}'
    tag_att = f'-att{args.att_out_dim}'
    tag_dropout = f'-dor{args.dropout}'
    tag_mbd = '-mbd' if args.use_multi_band else ''
    tag_sub = f'-sub{subject_id}'
    tag_data_ratio = f'-ratio{ratio}'
    if suffix:
        tag_data_ratio += f'-{suffix}'
    tag_emb = args.feature_name
    tag_valid = f'-valid{args.valid}'
    tag_seed = f'-seed{args.seed}'
    model_dir_name = (f'{args.model_name}-bs{args.batch_size}-sl{args.seg_len}{tag_mbd}'
                      f'-ks{args.kernel_size}{tag_dropout}{tag_att}{tag_nn}{tag_valid}'
                      f'-{tag_emb}{tag_sub}{tag_data_ratio}{tag_seed}')
    return model_dir_name


def get_ckpt_path(args, subject_id, ratio, suffix):
    model_dir_name = build_model_tag(args, subject_id, ratio, suffix=suffix)
    ckpt_path = op.join(result_v4_dir, 'models', model_dir_name, 'best', 'model.pt')
    return ckpt_path, model_dir_name


def get_pretrained_ckpt_path(args, subject_id, transfer):
    suffix = TRANSFER_SCENARIOS[transfer]['source_suffix']
    return get_ckpt_path(args, subject_id, 1.0, suffix)


# =========================================================================
# v3: 结果保存到 finetune_result_v3 (修改 finetune_result_dir 指向)
# =========================================================================
finetune_result_v3_dir = op.join(project_dir, 'finetune_result_v3')


def setup_logging_and_dirs(args, subject_id, ratio, strategy, transfer):
    strategy_tag = f'-{transfer}-S{strategy}'
    model_dir_name = build_model_tag(args, subject_id, ratio, suffix=f'finetune{strategy_tag}')
    print(f'model dir: {model_dir_name}')

    checkpoint_dir = op.join(finetune_result_v3_dir, 'models', model_dir_name)
    checkpoint_best_dir = op.join(checkpoint_dir, 'best')
    if not op.exists(checkpoint_best_dir):
        os.makedirs(checkpoint_best_dir)

    output_checkpoint_name_best = op.join(checkpoint_best_dir, 'model.pt')
    tensorboard_dir = op.join(checkpoint_dir, 'runs')
    if not op.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    file_handler = logging.FileHandler(op.join(checkpoint_dir, 'acc_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(model_dir_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('\n\n')
    logger.info('*****************************START NEW SESSION*****************************\n')
    logger.info(f'TRANSFER: {transfer} | STRATEGY: {strategy} | SUBJECT: {subject_id} | RATIO: {ratio}')
    logger.info(f'v3: common channels mode, in_dim = {NUM_COMMON_CHANNELS}')

    return logger, checkpoint_dir, checkpoint_best_dir, output_checkpoint_name_best, tensorboard_dir


# =========================================================================
# 数据加载
# =========================================================================
def load_stimulus_features(args):
    feature_dim_dict = DEFAULTS['feature_dim_dict']
    feature_name = args.feature_name
    seg_len = args.seg_len

    # 兼容 mel_10: 如果 feature_name 本身是 feature_dim_dict 里的已知特征名
    # (含下划线), 则不再拆分; 只有未知的组合名 (如 mel_bert) 才按 '_' 拆分。
    if '_' not in feature_name or feature_name in feature_dim_dict:
        feature_name_ls = [feature_name]
    else:
        feature_name_ls = feature_name.split('_')
    feature_name_ls_new = []
    for fn in feature_name_ls:
        if fn == 'wav2vec11layers':
            feature_name_ls_new.extend([f'wav2vec{layer}pca32' for layer in range(2, 24, 2)])
        else:
            feature_name_ls_new.append(fn)
    feature_name_ls = feature_name_ls_new

    out_dim_ls = [feature_dim_dict[fn] for fn in feature_name_ls]
    out_dim = sum(out_dim_ls)

    feat_dict_ls = []
    feat_frames_ls = []
    for feat in feature_name_ls:
        if feat == 'envelope':
            feat_fs = 100
        else:
            feat_fs = 50
        feat_dict_1, feat_frames_id_dict_1, feat_keys_1, feat_frames_1 = \
            get_stimulus_feat(feature_name=feat, seg_len=seg_len, fs=feat_fs)
        feat_dict_ls.append(feat_dict_1)
        feat_frames_ls.append(feat_frames_1)

    feat_dict = {}
    for key in feat_keys_1:
        feat_dict[key] = np.concatenate([fd[key] for fd in feat_dict_ls], axis=-1)
    feat_frames = np.concatenate(feat_frames_ls, axis=-1)
    feat_frames_id_dict = feat_frames_id_dict_1
    feat_keys = feat_keys_1

    embeds = torch.tensor(feat_frames, dtype=torch.float32)
    feat_frames_id = [f for f in feat_frames_id_dict.values()]

    return feat_dict, feat_frames_id_dict, feat_keys, embeds, feat_frames_id, out_dim


def load_domain_data(subject_id, feat_dict, feat_frames_id_dict, feat_keys,
                     story_range, eeg_device, seg_len, fs, use_multi_band,
                     feature_name='envelope', description='data'):
    sub_ls = [subject_id]
    sub_names = ['sub-{:02d}'.format(i) for i in range(1, 26)]

    EEG_dict, EEG_idx, Sub_dict, Sti_dict, id2dict, _ = get_EEG_emb_feat_story_range(
        sub_ls, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=story_range, eeg_device=eeg_device,
        seg_len=seg_len, fs=fs, std_eeg='std', phase=description,
        feature_name=feature_name
    )

    if len(id2dict) == 0:
        raise ValueError(f"[ERROR] No data found for subject {subject_id} "
                         f"with eeg_device={eeg_device}, story_range={story_range}")

    dataset = KUL_dataset(EEG_dict, EEG_idx, Sub_dict, Sti_dict, id2dict,
                          sub_names, use_multi_band=use_multi_band)
    return dataset, EEG_dict, EEG_idx, Sub_dict, Sti_dict, id2dict


def load_data_for_strategy(args, subject_id, ratio, transfer):
    seg_len = args.seg_len
    fs = args.fs
    use_multi_band = args.use_multi_band
    feature_name = args.feature_name
    scenario = TRANSFER_SCENARIOS[transfer]

    feat_dict, feat_frames_id_dict, feat_keys, embeds, feat_frames_id, out_dim = \
        load_stimulus_features(args)

    src_cfg = scenario['source']
    src_train_set, _, _, _, _, _ = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=src_cfg['train'], eeg_device=src_cfg['device'],
        seg_len=seg_len, fs=fs, use_multi_band=use_multi_band,
        feature_name=feature_name, description='source-train'
    )
    src_val_set, _, _, _, _, _ = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=src_cfg['valid'], eeg_device=src_cfg['device'],
        seg_len=seg_len, fs=fs, use_multi_band=use_multi_band,
        feature_name=feature_name, description='source-valid'
    )

    tgt_cfg = scenario['target']
    tgt_train_full_set, _, _, _, _, tgt_train_id2dict = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=tgt_cfg['train'], eeg_device=tgt_cfg['device'],
        seg_len=seg_len, fs=fs, use_multi_band=use_multi_band,
        feature_name=feature_name, description='target-train'
    )

    total_target = len(tgt_train_full_set)
    selected_count = max(1, int(total_target * ratio))
    train_idxs = list(range(selected_count))
    sub_names = ['sub-{:02d}'.format(i) for i in range(1, 26)]
    tgt_train_set = KUL_dataset(
        tgt_train_full_set.EEG_dict,
        tgt_train_full_set.EEG_feat_index_dict,
        tgt_train_full_set.Sub_id_dict,
        tgt_train_full_set.Stimulus_index_dict,
        tgt_train_id2dict,
        sub_names,
        use_multi_band=use_multi_band,
        indices=train_idxs
    )

    tgt_val_set, _, _, _, _, _ = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=tgt_cfg['valid'], eeg_device=tgt_cfg['device'],
        seg_len=seg_len, fs=fs, use_multi_band=use_multi_band,
        feature_name=feature_name, description='target-valid'
    )
    tgt_test_set, _, _, _, _, _ = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=tgt_cfg['test'], eeg_device=tgt_cfg['device'],
        seg_len=seg_len, fs=fs, use_multi_band=use_multi_band,
        feature_name=feature_name, description='target-test'
    )

    print(f"\nData loaded for transfer '{transfer}', subject {subject_id}, ratio {ratio}:")
    print(f"  Source train: {len(src_train_set)} segments  (device={src_cfg['device']})")
    print(f"  Source valid: {len(src_val_set)} segments")
    print(f"  Target train: {len(tgt_train_set)} / {total_target} segments  (device={tgt_cfg['device']})")
    print(f"  Target valid: {len(tgt_val_set)} segments")
    print(f"  Target test:  {len(tgt_test_set)} segments")

    return (feat_dict, feat_frames_id_dict, feat_keys, embeds, feat_frames_id, out_dim,
            src_train_set, src_val_set, tgt_train_set, tgt_val_set, tgt_test_set)


# =========================================================================
# 模型构建
# =========================================================================
def build_model(args, out_dim, embeds, feat_frames_id, device, sub_ls_all):
    # v3: in_dim 固定为公共通道数
    in_dim = NUM_COMMON_CHANNELS
    if args.use_multi_band:
        in_dim = 4 * in_dim
    print(f'[INFO] in_dim (common channels): {in_dim} (base={NUM_COMMON_CHANNELS})')

    model = BrainNetworkCL(
        in_dim=in_dim, att_out_dim=args.att_out_dim, out_dim=out_dim,
        dropout=args.dropout, train_subs=sub_ls_all,
        negsample_num=args.negsample_num, device=device,
        kernel_size=args.kernel_size, embeds=embeds, feat_frames_id=feat_frames_id
    )
    model.to(device)
    return model


def load_pretrained_model(ckpt_path, args, out_dim, embeds, feat_frames_id, device, sub_ls_all):
    print(f"Loading pretrained model from: {ckpt_path}")
    model = build_model(args, out_dim, embeds, feat_frames_id, device, sub_ls_all)
    checkpoint = torch.load(ckpt_path, map_location=device)
    # v3 note: 如果预训练模型也用公共通道训练 (v4), 则维度匹配 (in_dim=57)。
    # 如果预训练模型来自 v3 (in_dim=56 or 64), 加载会因维度不匹配而失败。
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Pretrained model loaded. (epoch={checkpoint.get('epoch', '?')}, "
          f"top1_acc={checkpoint.get('top1_acc', '?')})")
    return model


# =========================================================================
# 五种策略实现 (与 v2 相同, 仅 build_model 调用的 in_dim 变了)
# =========================================================================

def strategy_1_zero_shot(args, subject_id, ratio, transfer, device, logger, writer,
                          checkpoint_dir, output_checkpoint_name_best):
    logger.info("=" * 60)
    logger.info(f"STRATEGY 1: Zero-shot ({TRANSFER_SCENARIOS[transfer]['name']})")
    logger.info("=" * 60)

    _, _, _, embeds, feat_frames_id, out_dim, \
        _, _, _, _, tgt_test_set = \
        load_data_for_strategy(args, subject_id, ratio, transfer)

    sub_ls_all = [subject_id]

    ckpt_path, _ = get_pretrained_ckpt_path(args, subject_id, transfer)
    if not op.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    model = load_pretrained_model(ckpt_path, args, out_dim, embeds, feat_frames_id, device, sub_ls_all)

    valid_batch = 32
    test_dataloader = DataLoader(tgt_test_set, batch_size=valid_batch, shuffle=False,
                                 num_workers=args.num_workers)
    dataloaders = {'valid': test_dataloader}

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    logger.info("Zero-shot evaluation on target test set...")
    test_acc, test_acc_sub_dict = valid_model(
        model, dataloaders, device, optimizer, embeds, phase='valid'
    )

    logger.info(f"Zero-shot Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    results = {
        'transfer': transfer,
        'strategy': 1,
        'strategy_name': 'Zero-shot Historical Model',
        'subject_id': subject_id,
        'ratio': ratio,
        'test_accuracy': test_acc,
        'per_subject_test_accuracy': test_acc_sub_dict,
        'seed': args.seed,
        'in_dim': NUM_COMMON_CHANNELS,
    }
    results_path = op.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")

    return test_acc, test_acc_sub_dict


def strategy_2_pretrain_finetune(args, subject_id, ratio, transfer, device, logger, writer,
                                  checkpoint_dir, output_checkpoint_name_best):
    logger.info("=" * 60)
    logger.info(f"STRATEGY 2: Pre-train + Target Fine-tuning ({TRANSFER_SCENARIOS[transfer]['name']})")
    logger.info("=" * 60)

    _, _, _, embeds, feat_frames_id, out_dim, \
        _, _, tgt_train_set, tgt_val_set, tgt_test_set = \
        load_data_for_strategy(args, subject_id, ratio, transfer)

    sub_ls_all = [subject_id]

    ckpt_path, _ = get_pretrained_ckpt_path(args, subject_id, transfer)
    if not op.exists(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    model = load_pretrained_model(ckpt_path, args, out_dim, embeds, feat_frames_id, device, sub_ls_all)

    batch_size = args.batch_size
    valid_batch = 32
    train_dataloader = DataLoader(tgt_train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=seed_worker, generator=build_generator(args.seed))
    val_dataloader = DataLoader(tgt_val_set, batch_size=valid_batch, shuffle=False,
                                num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=build_generator(args.seed + 1))
    test_dataloader = DataLoader(tgt_test_set, batch_size=valid_batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 worker_init_fn=seed_worker, generator=build_generator(args.seed + 2))

    dataloaders = {'train': train_dataloader, 'valid': val_dataloader, 'valid_all': test_dataloader}
    dataset_sizes = {'train': len(tgt_train_set), 'valid': len(tgt_val_set), 'test': len(tgt_test_set)}

    ft_lr = args.lr
    ft_epochs = 50
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=ft_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=ft_lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    logger.info(f'Starting fine-tuning from pretrained weights (lr={ft_lr}, epochs={ft_epochs})...')
    model, best_top1, best_epoch = train_model(
        dataloaders, dataset_sizes, device, model,
        optimizer=optimizer, scheduler=scheduler,
        start_epoch=0, num_epochs=ft_epochs,
        best_top1=None, negsample_num=args.negsample_num,
        embeds=embeds, checkpoint_path_best=output_checkpoint_name_best,
        early_stop_num=args.early_stop_num, logger=logger, writer=writer,
        seed=args.seed
    )

    checkpoint = torch.load(output_checkpoint_name_best)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_acc_sub_dict = valid_model(
        model, dataloaders, device, optimizer, embeds, phase='valid_all'
    )

    logger.info(f"Fine-tuned Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    results = {
        'transfer': transfer,
        'strategy': 2,
        'strategy_name': 'Pre-train + Target Fine-tuning',
        'subject_id': subject_id,
        'ratio': ratio,
        'best_validation_epoch': best_epoch,
        'best_validation_accuracy': best_top1,
        'test_accuracy': test_acc,
        'per_subject_test_accuracy': test_acc_sub_dict,
        'seed': args.seed,
        'in_dim': NUM_COMMON_CHANNELS,
    }
    results_path = op.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")

    return test_acc, test_acc_sub_dict


def strategy_3_target_only(args, subject_id, ratio, transfer, device, logger, writer,
                            checkpoint_dir, output_checkpoint_name_best):
    logger.info("=" * 60)
    logger.info(f"STRATEGY 3: Target-only Eval ({TRANSFER_SCENARIOS[transfer]['name']})")
    logger.info(f"  Loading existing target model from result_v4, ratio={ratio}")

    feat_dict, feat_frames_id_dict, feat_keys, embeds, feat_frames_id, out_dim = \
        load_stimulus_features(args)

    tgt_cfg = TRANSFER_SCENARIOS[transfer]['target']
    tgt_test_set, _, _, _, _, _ = load_domain_data(
        subject_id, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=tgt_cfg['test'], eeg_device=tgt_cfg['device'],
        seg_len=args.seg_len, fs=args.fs, use_multi_band=args.use_multi_band,
        feature_name=args.feature_name, description='target-test'
    )
    logger.info(f"  Target test: {len(tgt_test_set)} segments")

    target_suffix = TRANSFER_SCENARIOS[transfer]['target_suffix']
    ckpt_path, src_model_name = get_ckpt_path(args, subject_id, ratio, target_suffix)
    if not op.exists(ckpt_path):
        raise FileNotFoundError(f"Target model checkpoint not found: {ckpt_path}")
    logger.info(f"  Loading model: {src_model_name}")

    sub_ls_all = [subject_id]
    model = load_pretrained_model(ckpt_path, args, out_dim, embeds, feat_frames_id, device, sub_ls_all)

    valid_batch = 32
    test_dataloader = DataLoader(tgt_test_set, batch_size=valid_batch, shuffle=False,
                                 num_workers=args.num_workers)
    dataloaders = {'valid': test_dataloader}

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    logger.info("Evaluating target-only model on target test set...")
    test_acc, test_acc_sub_dict = valid_model(
        model, dataloaders, device, optimizer, embeds, phase='valid'
    )

    logger.info(f"Target-only Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    results = {
        'transfer': transfer,
        'strategy': 3,
        'strategy_name': 'Target-only Model (from result_v4)',
        'subject_id': subject_id,
        'ratio': ratio,
        'source_model': src_model_name,
        'test_accuracy': test_acc,
        'per_subject_test_accuracy': test_acc_sub_dict,
        'seed': args.seed,
        'in_dim': NUM_COMMON_CHANNELS,
    }
    results_path = op.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")

    return test_acc, test_acc_sub_dict


def strategy_4_target_then_historical(args, subject_id, ratio, transfer, device, logger, writer,
                                       checkpoint_dir, output_checkpoint_name_best):
    logger.info("=" * 60)
    logger.info(f"STRATEGY 4: Target-train + Historical Fine-tuning ({TRANSFER_SCENARIOS[transfer]['name']})")
    logger.info("=" * 60)

    _, _, _, embeds, feat_frames_id, out_dim, \
        src_train_set, src_val_set, tgt_train_set, tgt_val_set, tgt_test_set = \
        load_data_for_strategy(args, subject_id, ratio, transfer)

    sub_ls_all = [subject_id]
    valid_batch = 32

    logger.info("Phase 1: Training from scratch on target data...")
    model = build_model(args, out_dim, embeds, feat_frames_id, device, sub_ls_all)

    train_dataloader = DataLoader(tgt_train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=seed_worker, generator=build_generator(args.seed))
    val_dataloader = DataLoader(tgt_val_set, batch_size=valid_batch, shuffle=False,
                                num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=build_generator(args.seed + 1))
    dataloaders_phase1 = {'train': train_dataloader, 'valid': val_dataloader}
    dataset_sizes_phase1 = {'train': len(tgt_train_set), 'valid': len(tgt_val_set)}

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    phase1_ckpt = output_checkpoint_name_best.replace('.pt', '_phase1.pt')
    model, best_top1_p1, best_epoch_p1 = train_model(
        dataloaders_phase1, dataset_sizes_phase1, device, model,
        optimizer=optimizer, scheduler=scheduler,
        start_epoch=0, num_epochs=args.num_epochs,
        best_top1=None, negsample_num=args.negsample_num,
        embeds=embeds, checkpoint_path_best=phase1_ckpt,
        early_stop_num=args.early_stop_num, logger=logger, writer=writer,
        seed=args.seed
    )
    logger.info(f"Phase 1 completed. Best val acc: {best_top1_p1:.4f} at epoch {best_epoch_p1}")

    logger.info("Phase 2: Fine-tuning on source (historical) data...")
    checkpoint_p1 = torch.load(phase1_ckpt)
    model.load_state_dict(checkpoint_p1["model_state_dict"])

    hist_train_dataloader = DataLoader(src_train_set, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers,
                                       worker_init_fn=seed_worker, generator=build_generator(args.seed + 100))
    hist_val_dataloader = DataLoader(src_val_set, batch_size=valid_batch, shuffle=False,
                                     num_workers=args.num_workers,
                                     worker_init_fn=seed_worker, generator=build_generator(args.seed + 101))
    test_dataloader = DataLoader(tgt_test_set, batch_size=valid_batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 worker_init_fn=seed_worker, generator=build_generator(args.seed + 102))

    dataloaders_phase2 = {'train': hist_train_dataloader, 'valid': hist_val_dataloader,
                          'valid_all': test_dataloader}
    dataset_sizes_phase2 = {'train': len(src_train_set), 'valid': len(src_val_set),
                            'test': len(tgt_test_set)}

    ft_lr = args.lr * 0.1
    if args.opt == 'Adam':
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=ft_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=ft_lr, momentum=0.9)
    scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.1)

    phase2_epochs = 50
    logger.info(f'Phase 2 fine-tuning with lr={ft_lr}, epochs={phase2_epochs}')

    model, best_top1_p2, best_epoch_p2 = train_model(
        dataloaders_phase2, dataset_sizes_phase2, device, model,
        optimizer=optimizer2, scheduler=scheduler2,
        start_epoch=0, num_epochs=phase2_epochs,
        best_top1=None, negsample_num=args.negsample_num,
        embeds=embeds, checkpoint_path_best=output_checkpoint_name_best,
        early_stop_num=args.early_stop_num, logger=logger, writer=writer,
        seed=args.seed
    )
    logger.info(f"Phase 2 completed. Best val acc: {best_top1_p2:.4f} at epoch {best_epoch_p2}")

    checkpoint = torch.load(output_checkpoint_name_best)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_dataloaders = {'valid_all': test_dataloader}
    test_acc, test_acc_sub_dict = valid_model(
        model, final_dataloaders, device,
        torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr),
        embeds, phase='valid_all'
    )

    logger.info(f"Strategy 4 Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    results = {
        'transfer': transfer,
        'strategy': 4,
        'strategy_name': 'Target-train + Historical Fine-tuning',
        'subject_id': subject_id,
        'ratio': ratio,
        'phase1_best_epoch': best_epoch_p1,
        'phase1_best_val_accuracy': best_top1_p1,
        'phase2_best_epoch': best_epoch_p2,
        'phase2_best_val_accuracy': best_top1_p2,
        'test_accuracy': test_acc,
        'per_subject_test_accuracy': test_acc_sub_dict,
        'seed': args.seed,
        'in_dim': NUM_COMMON_CHANNELS,
    }
    results_path = op.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")

    return test_acc, test_acc_sub_dict


def strategy_5_joint_training(args, subject_id, ratio, transfer, device, logger, writer,
                                checkpoint_dir, output_checkpoint_name_best):
    logger.info("=" * 60)
    logger.info(f"STRATEGY 5: Joint Training ({TRANSFER_SCENARIOS[transfer]['name']})")
    logger.info("=" * 60)

    _, _, _, embeds, feat_frames_id, out_dim, \
        src_train_set, src_val_set, tgt_train_set, tgt_val_set, tgt_test_set = \
        load_data_for_strategy(args, subject_id, ratio, transfer)

    sub_ls_all = [subject_id]
    sub_name = 'sub-{:02d}'.format(subject_id)
    sub_idx = subject_id - 1
    sub_names_list = ['sub-{:02d}'.format(i) for i in range(1, 26)]

    def merge_datasets(set_a, set_b):
        eeg_a = set_a.EEG_dict[sub_name]
        eeg_b = set_b.EEG_dict[sub_name]
        joint_eeg = np.concatenate([eeg_a, eeg_b], axis=0)
        joint_eeg_dict = {sub_name: joint_eeg}

        feat_a = set_a.EEG_feat_index_dict[sub_name]
        feat_b = set_b.EEG_feat_index_dict[sub_name]
        joint_feat = np.concatenate([feat_a, feat_b], axis=0)
        joint_feat_dict = {sub_name: joint_feat}

        sub_a = set_a.Sub_id_dict[sub_name]
        sub_b = set_b.Sub_id_dict[sub_name]
        joint_sub = np.concatenate([sub_a, sub_b], axis=0)
        joint_sub_dict = {sub_name: joint_sub}

        sti_a = set_a.Stimulus_index_dict[sub_name]
        sti_b = set_b.Stimulus_index_dict[sub_name]
        joint_sti = np.concatenate([sti_a, sti_b], axis=0)
        joint_sti_dict = {sub_name: joint_sti}

        joint_id2dict = np.array([[sub_idx, i] for i in range(len(joint_eeg))])
        return KUL_dataset(joint_eeg_dict, joint_feat_dict, joint_sub_dict, joint_sti_dict,
                           joint_id2dict, sub_names_list, use_multi_band=args.use_multi_band)

    joint_train_set = merge_datasets(src_train_set, tgt_train_set)
    print(f"Joint training set: {len(joint_train_set)} segments "
          f"(source: {len(src_train_set)}, target: {len(tgt_train_set)})")
    joint_val_set = merge_datasets(src_val_set, tgt_val_set)
    print(f"Joint validation set: {len(joint_val_set)} segments "
          f"(source: {len(src_val_set)}, target: {len(tgt_val_set)})")

    batch_size = args.batch_size
    valid_batch = 32
    train_dataloader = DataLoader(joint_train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  worker_init_fn=seed_worker, generator=build_generator(args.seed))
    val_dataloader = DataLoader(joint_val_set, batch_size=valid_batch, shuffle=False,
                                num_workers=args.num_workers,
                                worker_init_fn=seed_worker, generator=build_generator(args.seed + 1))
    test_dataloader = DataLoader(tgt_test_set, batch_size=valid_batch, shuffle=False,
                                 num_workers=args.num_workers,
                                 worker_init_fn=seed_worker, generator=build_generator(args.seed + 2))

    dataloaders = {'train': train_dataloader, 'valid': val_dataloader, 'valid_all': test_dataloader}
    dataset_sizes = {'train': len(joint_train_set), 'valid': len(joint_val_set), 'test': len(tgt_test_set)}

    model = build_model(args, out_dim, embeds, feat_frames_id, device, sub_ls_all)

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    logger.info('Joint training from scratch...')
    model, best_top1, best_epoch = train_model(
        dataloaders, dataset_sizes, device, model,
        optimizer=optimizer, scheduler=scheduler,
        start_epoch=0, num_epochs=args.num_epochs,
        best_top1=None, negsample_num=args.negsample_num,
        embeds=embeds, checkpoint_path_best=output_checkpoint_name_best,
        early_stop_num=args.early_stop_num, logger=logger, writer=writer,
        seed=args.seed
    )

    checkpoint = torch.load(output_checkpoint_name_best)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_acc_sub_dict = valid_model(
        model, dataloaders, device, optimizer, embeds, phase='valid_all'
    )

    logger.info(f"Joint Training Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    results = {
        'transfer': transfer,
        'strategy': 5,
        'strategy_name': 'Joint Training',
        'subject_id': subject_id,
        'ratio': ratio,
        'best_validation_epoch': best_epoch,
        'best_validation_accuracy': best_top1,
        'test_accuracy': test_acc,
        'per_subject_test_accuracy': test_acc_sub_dict,
        'seed': args.seed,
        'in_dim': NUM_COMMON_CHANNELS,
    }
    results_path = op.join(checkpoint_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")

    return test_acc, test_acc_sub_dict


# =========================================================================
# 策略调度器
# =========================================================================
STRATEGY_FUNCTIONS = {
    1: strategy_1_zero_shot,
    2: strategy_2_pretrain_finetune,
    3: strategy_3_target_only,
    4: strategy_4_target_then_historical,
    5: strategy_5_joint_training,
}

STRATEGY_NAMES = {
    1: 'Zero-shot Historical Model',
    2: 'Pre-train + Target Fine-tuning',
    3: 'Target-only Training from Scratch',
    4: 'Target-train + Historical Fine-tuning',
    5: 'Joint Training',
}


def run_strategy(args):
    strategy = args.strategy
    subject_id = args.subject_id
    ratio = args.data_ratio
    transfer = args.transfer
    device = args.device

    if not torch.cuda.is_available():
        device = "cpu"
    print(f'[INFO] using device {device}')

    if strategy not in STRATEGY_FUNCTIONS:
        raise ValueError(f"Unknown strategy {strategy}")
    if transfer not in TRANSFER_SCENARIOS:
        raise ValueError(f"Unknown transfer '{transfer}'. Choices: {list(TRANSFER_SCENARIOS.keys())}")

    logger, checkpoint_dir, _, output_checkpoint_name_best, tensorboard_dir = \
        setup_logging_and_dirs(args, subject_id, ratio, strategy, transfer)

    writer = SummaryWriter(tensorboard_dir)

    args_dict = vars(args)
    logger.info(f'ARGS: {json.dumps(args_dict, indent=4)}')

    strategy_fn = STRATEGY_FUNCTIONS[strategy]
    strategy_name = STRATEGY_NAMES[strategy]
    transfer_name = TRANSFER_SCENARIOS[transfer]['name']

    print(f"\n{'=' * 60}")
    print(f"Transfer: {transfer} ({transfer_name})")
    print(f"Strategy {strategy}: {strategy_name}")
    print(f"Subject: {subject_id}, Ratio: {ratio}, Device: {device}")
    print(f"{'=' * 60}\n")

    try:
        test_acc, test_acc_sub_dict = strategy_fn(
            args, subject_id, ratio, transfer, device, logger, writer,
            checkpoint_dir, output_checkpoint_name_best
        )

        print(f"\n{'=' * 60}")
        print(f"COMPLETED — Transfer: {transfer}, Strategy {strategy} ({strategy_name})")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"{'=' * 60}\n")

        writer.close()
        return test_acc, test_acc_sub_dict

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        writer.close()
        raise


# =========================================================================
# 命令行入口
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module B v3: Finetuning Strategies (Common Channels)')

    parser.add_argument('--model_name', default=DEFAULTS['model_name'])
    parser.add_argument('--seg_len', type=int, default=DEFAULTS['seg_len'])
    parser.add_argument('--fs', type=float, default=DEFAULTS['fs'])
    parser.add_argument('--device', default=DEFAULTS['device'])
    parser.add_argument('--opt', default=DEFAULTS['opt'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr_dic'][DEFAULTS['opt']])
    parser.add_argument('--num_epochs', type=int, default=DEFAULTS['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--negsample_num', type=int, default=DEFAULTS['negsample_num'])
    parser.add_argument('--kernel_size', type=int, default=DEFAULTS['kernel_size'])
    parser.add_argument('--feature_name', default=DEFAULTS['feature_name'],
                        help='Feature type: envelope, mel, bert, wav2vecbase9')
    parser.add_argument('--valid', type=int, default=DEFAULTS['valid'])
    parser.add_argument('--att_out_dim', type=int, default=DEFAULTS['att_out_dim'])
    parser.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    parser.add_argument('--use_multi_band', type=bool, default=DEFAULTS['use_multi_band'])
    parser.add_argument('--early_stop_num', type=int, default=DEFAULTS['early_stop_num'])
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'])

    parser.add_argument('--subject_id', type=int, required=True, help='Subject ID (1-25)')
    parser.add_argument('--transfer', type=str, required=True,
                        choices=list(TRANSFER_SCENARIOS.keys()))
    parser.add_argument('--strategy', type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--data_ratio', type=float, default=1.0)

    args = parser.parse_args()
    args.lr = DEFAULTS['lr_dic'][args.opt]

    seed_everything(args.seed)
    print(f'[INFO] random seed: {args.seed}')
    print(f'[INFO] common channels: {NUM_COMMON_CHANNELS}')

    run_strategy(args)
