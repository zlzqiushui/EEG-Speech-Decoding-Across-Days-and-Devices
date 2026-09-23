"""
train_models_sub_v4.py — 公共通道版本

与 train_models_sub_v3.py 的差异:
  - 使用 data_all_v2 (公共通道版本), 不再导入 data_all_v1。
  - in_dim 固定为 57 (三设备公共通道数), 不随 eeg_device 变化。
  - 所有 EEG 数据按 COMMON_CHANNELS (57 个, 标准 10-20 顺序) 对齐。
  - 结果保存到 result_v4。
"""

import random
import numpy as np
import config as cfg
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_all_v2 import (KUL_dataset, get_stimulus_feat, get_EEG_emb_feat_story_range,
                         NUM_COMMON_CHANNELS, COMMON_CHANNELS)
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
import copy
import os
import os.path as op
import json
from model_v1 import BrainNetworkCL


# =========================================================================
# 默认配置常量 (对齐 train_models_sub_v3.py)
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
    feature_name='mel',
    feature_dim_dict={'wav2vecbase9': 1024, 'bert': 768, 'envelope': 1, 'mel': 80, 'mel_10':10},
    valid=1,
    seed=42,
    num_workers=16,
    eeg_device='neurascan'  # 'brk' 'both' 'neurascan' — 仅影响故事范围, 不影响通道维度
)


def str2bool(v):
    """argparse 友好的布尔参数解析。"""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    if v in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def seed_everything(seed=42):
    """固定 Python / NumPy / PyTorch / CUDA 的随机种子，保证训练可复现。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """让 DataLoader 的每个 worker 使用可复现的 NumPy / random 种子。"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed):
    """为 DataLoader 构造独立的 torch.Generator。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def valid_model(model, dataloaders, device, optimizer, embeds, phase='valid'):
    model.eval()
    epoch_induce = None
    sub_ids = []

    for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(tqdm(dataloaders[phase]), start=1):
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
    for sub_id in sub_ids_set:
        sub_name = sub_names[sub_id]
        acc_sub = acc[sub_ids == sub_id]
        acc_sub = acc_sub.sum().item() / len(acc_sub)
        acc_sub_dict[sub_name] = acc_sub

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
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        running_loss = 0.0
        epoch_rank = None

        for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(tqdm(dataloaders[phase]), start=1):
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
            it_num = 200

            if global_step % it_num == 0:
                top10_iter50 = torch.nonzero(iter50_rank <= 10).shape[0] / iter50_rank.shape[0]
                top5_iter50 = torch.nonzero(iter50_rank <= 5).shape[0] / iter50_rank.shape[0]
                top1_iter50 = torch.nonzero(iter50_rank == 1).shape[0] / iter50_rank.shape[0]
                sample_size = negsample_num + 1
                rank_acc_iter50 = (sample_size - torch.mean(iter50_rank.float())) / (sample_size - 1)

                writer.add_scalar('Iter50/Loss', iter50_loss / it_num, global_step // it_num)
                writer.add_scalars('Iter50/Accuracy',
                                   {'top-10': top10_iter50,
                                    'top-5': top5_iter50,
                                    'top-1': top1_iter50,
                                    'rank-acc': rank_acc_iter50},
                                   global_step // it_num)

                logger.info("Train: epoch {:.0f}/ global_step {:.0f}: loss: {:.4f}; top-10: {:.4f}; top-5: {:.4f}; top-1: {:.4f}; rank-acc: {:.4f}"
                            .format(epoch, global_step, iter50_loss / it_num, top10_iter50, top5_iter50, top1_iter50, rank_acc_iter50))
                print('{} Loss: {:.4f}; top-10: {:.4f}; rank-acc: {:.4f}'
                      .format(phase, iter50_loss / it_num, top10_iter50, rank_acc_iter50))

                print('evaluating valid set...')
                top1_iter50_valid, acc_sub_dict = \
                    valid_model(model, dataloaders, device, optimizer, embeds, phase='valid')
                model.train()

                writer.add_scalars('Iter50/valid-Accuracy', {'top-1': top1_iter50_valid}, global_step // it_num)
                writer.add_scalars('Iter50/valid-Accuracy-sub', acc_sub_dict, global_step // it_num)

                logger.info("Dev: epoch {:.0f}/ global_step {:.0f}: top-1: {:.4f}; "
                            .format(epoch, global_step, top1_iter50_valid))
                acc_sub_dict_as_str = ', '.join([f'{k}: {v:.4f}' for k, v in acc_sub_dict.items()])
                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: acc_sub_dict: {acc_sub_dict_as_str}")
                print('valid top-1: {:.4f}'.format(top1_iter50_valid))

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
                    print(f'update best on valid checkpoint: {checkpoint_path_best}')
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
                           {'top-10': top10_epoch,
                            'top-5': top5_epoch,
                            'top-1': top1_epoch,
                            'rank-acc': rank_acc_epoch},
                           epoch)
        logger.info(" epoch {:.0f}: loss/{}: {:.5f}; top-10: {:.4f}; top-5: {:.4f}; top-1: {:.4f}; rank-acc: {:.4f}"
                    .format(epoch, phase, epoch_loss, top10_epoch, top5_epoch, top1_epoch, rank_acc_epoch))
        print('{} Loss: {:.4f}; top-10: {:.4f}; rank-acc: {:.4f}'
              .format(phase, epoch_loss, top10_epoch, rank_acc_epoch))

        if un_update_step > early_stop_num:
            print(f'early stop at epoch {epoch}, global_step {global_step}')
            logger.info(f'early stop at epoch {epoch}, global_step {global_step}')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc epoch {:0f}: {:4f}'.format(best_epoch, best_top1))
    logger.info('Best val loss epoch {:0f}: {:4f}'.format(best_epoch, best_top1))

    model.load_state_dict(best_model_wts)
    return model, best_top1, best_epoch


def train(args):
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    model_name = args.model_name
    device = args.device
    valid_batch = 32
    opt = args.opt
    seg_len = args.seg_len
    negsample_num = args.negsample_num
    kernel_size = args.kernel_size
    feature_name = args.feature_name
    valid = args.valid
    att_out_dim = args.att_out_dim
    use_multi_band = args.use_multi_band
    dropout = args.dropout
    early_stop_num = args.early_stop_num
    seed = args.seed
    num_workers = args.num_workers

    seed_everything(seed)
    print(f'[INFO]random seed: {seed}')
    print(f'[INFO]common channels: {NUM_COMMON_CHANNELS} channels')
    print(f'[INFO]channels: {COMMON_CHANNELS}')

    subject_id = args.subject_id
    data_ratio = args.data_ratio
    eeg_device = args.eeg_device

    # 标签生成
    tag_nn = f'-nn{negsample_num}'
    tag_att = f'-att{att_out_dim}'
    tag_dropout = f'-dor{dropout}'
    tag_mbd = '-mbd' if use_multi_band else ''
    tag_sub = f'-sub{subject_id}'
    tag_data_ratio = f'-ratio{data_ratio}'
    tag_seed = f'-seed{seed}'

    if eeg_device != 'both':
        tag_data_ratio += f'-{eeg_device}'

    # ---- 故事范围设置 (仅区分设备, 不影响通道维度) ----
    if eeg_device == 'brk':
        args.train_story_range_start = 34
        args.train_story_range_end = 45
        args.valid_story_range_start = 46
        args.valid_story_range_end = 48
        args.test_story_range_start = 49
        args.test_story_range_end = 50
        args.train_story_range = list(range(args.train_story_range_start, args.train_story_range_end + 1))
        args.valid_story_range = list(range(args.valid_story_range_start, args.valid_story_range_end + 1))
        args.test_story_range = list(range(args.test_story_range_start, args.test_story_range_end + 1))
    elif eeg_device == 'neurascan':
        args.train_story_range_start = 1
        args.train_story_range_end = 15
        args.valid_story_range_start = 16
        args.valid_story_range_end = 17
        args.test_story_range_start = 18
        args.test_story_range_end = 33
        args.train_story_range = list(range(args.train_story_range_start, args.train_story_range_end + 1))
        args.valid_story_range = list(range(args.valid_story_range_start, args.valid_story_range_end + 1))
        args.test_story_range = list(range(args.test_story_range_start, args.test_story_range_end + 1))
    elif eeg_device == 'day1':
        args.train_story_range_start = 1
        args.train_story_range_end = 12
        args.valid_story_range_start = 13
        args.valid_story_range_end = 15
        args.test_story_range_start = 16
        args.test_story_range_end = 17
        args.train_story_range = list(range(args.train_story_range_start, args.train_story_range_end + 1))
        args.valid_story_range = list(range(args.valid_story_range_start, args.valid_story_range_end + 1))
        args.test_story_range = list(range(args.test_story_range_start, args.test_story_range_end + 1))
    elif eeg_device == 'day2':
        args.train_story_range_start = 18
        args.train_story_range_end = 29
        args.valid_story_range_start = 30
        args.valid_story_range_end = 31
        args.test_story_range_start = 32
        args.test_story_range_end = 33
        args.train_story_range = list(range(args.train_story_range_start, args.train_story_range_end + 1))
        args.valid_story_range = list(range(args.valid_story_range_start, args.valid_story_range_end + 1))
        args.test_story_range = list(range(args.test_story_range_start, args.test_story_range_end + 1))

    # 特征名称展开
    feature_dim_dict = DEFAULTS['feature_dim_dict']
    if '_' not in feature_name or feature_name in feature_dim_dict:
        feature_name_ls = [feature_name]
    else:
        feature_name_ls = feature_name.split('_')
    feature_name_ls_new = []
    for fn in feature_name_ls:
        if fn == 'wav2vec11layers':
            feature_name_new = [f'wav2vec{layer}pca32' for layer in range(2, 24, 2)]
            feature_name_ls_new.extend(feature_name_new)
        else:
            feature_name_ls_new.append(fn)
    feature_name_ls = feature_name_ls_new

    out_dim_ls = [feature_dim_dict[feature_name] for feature_name in feature_name_ls]
    out_dim = sum(out_dim_ls)

    tag_emb = feature_name
    tag_valid = f'-valid{valid}'

    # ---- v4: 结果保存到 result_v4 ----
    model_dir = f'{model_name}-bs{batch_size}-sl{seg_len}{tag_mbd}-ks{kernel_size}{tag_dropout}{tag_att}{tag_nn}{tag_valid}-{tag_emb}{tag_sub}{tag_data_ratio}{tag_seed}'
    print(f'model dir: {model_dir}')
    checkpoint_dir = op.join(cfg.project_dir, 'result_v4', 'models', model_dir)
    checkpoint_best_dir = op.join(checkpoint_dir, 'best')
    output_checkpoint_name_best = op.join(checkpoint_best_dir, 'model.pt')
    if not op.exists(checkpoint_best_dir):
        os.makedirs(checkpoint_best_dir)
    tensorboard_dir = op.join(checkpoint_dir, 'runs')
    if not op.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    file_handler = logging.FileHandler(op.join(checkpoint_dir, 'acc_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(model_dir)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info('\n\n')
    logger.info('*****************************START NEW SESSION*****************************\n')
    logger.info('PARAMETER ...')
    logger.info(f'v4: common channels mode, in_dim = {NUM_COMMON_CHANNELS}')
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)
    logger.info(args_json)

    if not torch.cuda.is_available():
        device = "cpu"
    print(f'[INFO]using device {device}')

    # ---- 加载 stimulus 特征 ----
    feat_dict_ls = []
    feat_frames_ls = []
    for feat in feature_name_ls:
        if feat == 'envelope':
            feat_fs = 100
        else:
            feat_fs = 50
        feat_dict_1, feat_frames_id_dict_1, feat_keys_1, feat_frames_1 = (
            get_stimulus_feat(feature_name=feat, seg_len=seg_len, fs=feat_fs))
        feat_dict_ls.append(feat_dict_1)
        feat_frames_ls.append(feat_frames_1)

    feat_dict = {}
    for key in feat_keys_1:
        feat_dict[key] = np.concatenate([feat_dict_1[key] for feat_dict_1 in feat_dict_ls], axis=-1)
    feat_frames = np.concatenate(feat_frames_ls, axis=-1)
    feat_frames_id_dict = feat_frames_id_dict_1
    feat_keys = feat_keys_1

    embeds = torch.tensor(feat_frames, dtype=torch.float32)
    feat_frames_id = [f for f in feat_frames_id_dict.values()]

    sub_names = ['sub-{:02d}'.format(i) for i in range(1, 26)]
    sub_ls_all = [subject_id]

    # ---------------------------------------------------------------------------------
    # 1. 加载测试集
    # ---------------------------------------------------------------------------------
    print(f"\n[1/3] Loading Test Set for Subject {subject_id}...")
    test_EEG, test_idx, test_sub, test_sti, test_id2dict, _ = get_EEG_emb_feat_story_range(
        sub_ls_all, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=args.test_story_range, eeg_device=eeg_device,
        seg_len=seg_len, fs=250, std_eeg='std', phase='test', feature_name=feature_name
    )
    test_set = KUL_dataset(test_EEG, test_idx, test_sub, test_sti, test_id2dict,
                           sub_names, use_multi_band=use_multi_band)

    # ---------------------------------------------------------------------------------
    # 2. 加载验证集
    # ---------------------------------------------------------------------------------
    print(f"\n[2/3] Loading Validation Set for Subject {subject_id}...")
    val_EEG, val_idx, val_sub, val_sti, val_id2dict, _ = get_EEG_emb_feat_story_range(
        sub_ls_all, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=args.valid_story_range, eeg_device=eeg_device,
        seg_len=seg_len, fs=250, std_eeg='std', phase='valid', feature_name=feature_name
    )
    val_set = KUL_dataset(val_EEG, val_idx, val_sub, val_sti, val_id2dict,
                          sub_names, use_multi_band=use_multi_band)

    # ---------------------------------------------------------------------------------
    # 3. 加载训练集池
    # ---------------------------------------------------------------------------------
    print(f"\n[3/3] Loading Training Pool for Subject {subject_id}...")
    train_EEG, train_idx, train_sub, train_sti, train_id2dict, _ = get_EEG_emb_feat_story_range(
        sub_ls_all, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=args.train_story_range, eeg_device=eeg_device,
        seg_len=seg_len, fs=250, std_eeg='std', phase='train', feature_name=feature_name
    )

    # ---------------------------------------------------------------------------------
    # 4. 根据 data_ratio 控制数据规模
    # ---------------------------------------------------------------------------------
    total_available = len(train_id2dict)

    if total_available == 0:
        raise ValueError(f"[ERROR] No training data available for Subject {subject_id}. "
                         f"Please check if the story range is correct for this device.")

    selected_count = int(total_available * data_ratio)
    selected_count = max(1, selected_count)

    train_idxs = list(range(selected_count))

    print(f"\nData scaling: Requested {data_ratio*100}% data.")
    print(f"Sampled Training segments: {len(train_idxs)} (Total available: {total_available})")

    train_set = KUL_dataset(train_EEG, train_idx, train_sub, train_sti, train_id2dict,
                            sub_names, use_multi_band=use_multi_band, indices=train_idxs)

    dataset_sizes = {'train': len(train_set), 'valid': len(val_set), 'test': len(test_set)}
    print(f"\nFinal Dataset sizes: {dataset_sizes}")

    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=build_generator(seed)
    )
    val_dataloader = DataLoader(
        val_set, batch_size=valid_batch, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=build_generator(seed + 1)
    )
    test_dataloader = DataLoader(
        test_set, batch_size=valid_batch, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=build_generator(seed + 2)
    )

    dataloaders = {'train': train_dataloader, 'valid': val_dataloader, 'valid_all': test_dataloader}

    # ---- v4: in_dim 固定为公共通道数 (57), 不随 eeg_device 变化 ----
    in_dim = NUM_COMMON_CHANNELS
    if use_multi_band:
        in_dim = 4 * in_dim

    print(f'[INFO] in_dim (common channels based): {in_dim} (base={NUM_COMMON_CHANNELS}, multi_band={use_multi_band})')

    if model_name == 'BrainNetworkCL':
        model = BrainNetworkCL(in_dim=in_dim,
                               att_out_dim=att_out_dim,
                               out_dim=out_dim,
                               dropout=dropout,
                               train_subs=sub_ls_all,
                               negsample_num=negsample_num,
                               device=device,
                               kernel_size=kernel_size,
                               embeds=embeds,
                               feat_frames_id=feat_frames_id)
    else:
        raise ValueError('model_name not supported')
    model.to(device)

    writer = SummaryWriter(tensorboard_dir)

    if opt == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_top1 = None
    start_epoch = 0
    logger.info('training from the beginning, random initialized')
    print('training from the beginning, random initialized')

    model, best_top1, best_epoch = train_model(
        dataloaders, dataset_sizes, device, model,
        optimizer=optimizer, scheduler=exp_lr_scheduler,
        start_epoch=start_epoch, num_epochs=num_epochs,
        best_top1=best_top1,
        negsample_num=negsample_num,
        embeds=embeds,
        checkpoint_path_best=output_checkpoint_name_best,
        early_stop_num=early_stop_num,
        logger=logger,
        writer=writer,
        seed=seed
    )

    print("\n" + "=" * 50)
    print("Evaluating best model on test set...")
    logger.info("Evaluating best model on test set...")

    checkpoint = torch.load(output_checkpoint_name_best)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_acc_sub_dict = valid_model(
        model, dataloaders, device, optimizer, embeds, phase='valid_all'
    )

    print(f"\nTest Results (using best validation model from epoch {best_epoch}):")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Per-subject Test Accuracy:")
    for sub_name, acc in test_acc_sub_dict.items():
        print(f"  {sub_name}: {acc:.4f}")

    logger.info(f"Test Results (using best validation model from epoch {best_epoch}):")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Per-subject Test Accuracy: {test_acc_sub_dict}")

    test_results = {
        "best_validation_epoch": best_epoch,
        "best_validation_accuracy": best_top1,
        "seed": seed,
        "num_workers": num_workers,
        "test_accuracy": test_acc,
        "per_subject_test_accuracy": test_acc_sub_dict,
        "in_dim": in_dim,
        "num_common_channels": NUM_COMMON_CHANNELS,
        "common_channels": COMMON_CHANNELS,
    }
    test_results_path = op.join(checkpoint_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Test results saved to {test_results_path}")
    logger.info(f"Test results saved to {test_results_path}")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--feature_name', default=DEFAULTS['feature_name'])
    parser.add_argument('--valid', type=int, default=DEFAULTS['valid'])
    parser.add_argument('--att_out_dim', type=int, default=DEFAULTS['att_out_dim'])
    parser.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    parser.add_argument('--use_multi_band', type=str2bool, default=DEFAULTS['use_multi_band'])
    parser.add_argument('--early_stop_num', type=int, default=DEFAULTS['early_stop_num'])
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'], help='Random seed for reproducible training')
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'],
                        help='Number of DataLoader workers')

    parser.add_argument('--subject_id', type=int, default=1, help='Subject ID to use (1-25)')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Proportion of training data to use (0.0 to 1.0)')

    parser.add_argument('--train_story_range_start', type=int, default=1, help='Start of training story range')
    parser.add_argument('--train_story_range_end', type=int, default=30, help='End of training story range')
    parser.add_argument('--valid_story_range_start', type=int, default=31, help='Start of validation story range')
    parser.add_argument('--valid_story_range_end', type=int, default=33, help='End of validation story range')
    parser.add_argument('--test_story_range_start', type=int, default=34, help='Start of testing story range')
    parser.add_argument('--test_story_range_end', type=int, default=50, help='End of testing story range')

    parser.add_argument('--eeg_device', default=DEFAULTS['eeg_device'], help='EEG device type (brk, neurascan, both, day1, day2)')
    args = parser.parse_args()
    args.lr = DEFAULTS['lr_dic'][args.opt]

    args.train_story_range = list(range(args.train_story_range_start, args.train_story_range_end + 1))
    args.valid_story_range = list(range(args.valid_story_range_start, args.valid_story_range_end + 1))
    args.test_story_range = list(range(args.test_story_range_start, args.test_story_range_end + 1))

    train(args)
