"""
test_sub_v2.py — 公共通道版本 (被试内测试)

与 test_sub_v1.py 的差异:
  - 使用 data_all_v2 (公共通道版本), 不再使用 data_all_v1。
  - in_dim 固定为 57 (三设备公共通道数), 不随 eeg_device 变化。
  - feature_dim_dict 对齐 train_models_sub_v4.py: {'wav2vecbase9': 1024, 'bert': 768, 'envelope': 1, 'mel': 80}
  - 结果保存到 result_v4, 模型也从 result_v4 加载。
  - 加载 stimulus 特征时根据特征类型使用不同的 fs (envelope→100, 其他→50)。
  - 不再定义 TARGET_CHANNELS, 改用 data_all_v2 中导入的 COMMON_CHANNELS。
  - both_to_brk 改为 neurascan_to_brk (跨天跨设备测试)。
"""

import numpy as np
import config as cfg
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_all_v2 import (KUL_dataset, get_stimulus_feat, get_EEG_emb_feat_story_range,
                         NUM_COMMON_CHANNELS, COMMON_CHANNELS)
import argparse
import os
import os.path as op
import json
from model_v1 import BrainNetworkCL


# =========================================================================
# 默认配置常量 (对齐 train_models_sub_v4.py)
# =========================================================================
DEFAULTS = dict(
    kernel_size=3,
    att_out_dim=256,
    dropout=0.5,
    model_name='BrainNetworkCL',
    batch_size=32,
    negsample_num=32,
    seg_len=5,
    fs=250,
    feature_name='envelope',
    feature_dim_dict={'wav2vecbase9': 1024, 'bert': 768, 'envelope': 1, 'mel': 80},
    valid=1,
)


def evaluate(model, dataloader, device, embeds):
    model.eval()
    epoch_induce = None
    sub_ids = []

    for eeg, emb_id, sub_id, sti_id in tqdm(dataloader, desc='Testing'):
        embed = embeds[emb_id]
        eeg, embed, sub_id = eeg.to(device), embed.to(device), sub_id.to(device)

        with torch.no_grad():
            pred_induce = model.test(x=eeg, sub_id=sub_id, frame_id=emb_id, embed=embed, sti_id=sti_id)

        if epoch_induce is None:
            epoch_induce = pred_induce
        else:
            epoch_induce = torch.cat((epoch_induce, pred_induce))
        sub_ids.append(sub_id)

    sub_ids = torch.cat(sub_ids)
    sub_ids_set = torch.unique(sub_ids).detach().cpu().numpy().tolist()
    sub_names = dataloader.dataset.sub_names
    acc = epoch_induce == 0
    acc_sub_dict = {}
    for sub_id in sub_ids_set:
        sub_name = sub_names[sub_id]
        acc_sub = acc[sub_ids == sub_id]
        acc_sub = acc_sub.sum().item() / len(acc_sub)
        acc_sub_dict[sub_name] = acc_sub

    acc_sub_mean = np.mean(list(acc_sub_dict.values()))
    return acc_sub_mean, acc_sub_dict


def test(args):
    device = args.device
    subject_id = args.subject_id
    data_ratio = args.data_ratio
    test_type = args.test_type

    batch_size = DEFAULTS['batch_size']
    seg_len = DEFAULTS['seg_len']
    kernel_size = DEFAULTS['kernel_size']
    att_out_dim = DEFAULTS['att_out_dim']
    dropout = DEFAULTS['dropout']
    negsample_num = DEFAULTS['negsample_num']
    feature_name = DEFAULTS['feature_name']
    valid = DEFAULTS['valid']
    model_name = DEFAULTS['model_name']

    # Determine source eeg_device and target settings based on test_type
    if test_type == 'day1_to_day2':
        # 模型用 day1 训练, 在同被试 day2 数据上测试
        source_eeg_device = 'day1'
        target_eeg_device = 'day2'
        test_story_range_start = 32
        test_story_range_end = 33
        output_filename = 'test_results_day2.json'
    elif test_type == 'day2_to_day1':
        # 模型用 day2 训练, 在同被试 day1 数据上测试
        source_eeg_device = 'day2'
        target_eeg_device = 'day1'
        test_story_range_start = 16
        test_story_range_end = 17
        output_filename = 'test_results_day1.json'
    elif test_type == 'neurascan_to_brk':
        # 跨天跨设备: neurascan (day1+day2) 训练的模型 → 同被试 brk 数据
        source_eeg_device = 'neurascan'
        target_eeg_device = 'brk'
        test_story_range_start = 49
        test_story_range_end = 50
        output_filename = 'test_results_brk.json'
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    # Build model directory tag (same as training script, v4)
    tag_nn = f'-nn{negsample_num}'
    tag_att = f'-att{att_out_dim}'
    tag_dropout = f'-dor{dropout}'
    tag_sub = f'-sub{subject_id}'
    tag_data_ratio = f'-ratio{data_ratio}'

    if source_eeg_device != 'both':
        tag_data_ratio += f'-{source_eeg_device}'

    tag_emb = feature_name
    tag_valid = f'-valid{valid}'

    # v2: 结果/模型路径改为 result_v4 (对齐 train_models_sub_v4.py)
    model_dir = f'{model_name}-bs{batch_size}-sl{seg_len}-ks{kernel_size}{tag_dropout}{tag_att}{tag_nn}{tag_valid}-{tag_emb}{tag_sub}{tag_data_ratio}'
    print(f'model dir: {model_dir}')
    checkpoint_dir = op.join(cfg.project_dir, 'result_v4', 'models', model_dir)
    checkpoint_path = op.join(checkpoint_dir, 'best', 'model.pt')

    if not op.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if not torch.cuda.is_available():
        device = "cpu"
    print(f'[INFO] using device {device}')
    print(f'[INFO] common channels: {NUM_COMMON_CHANNELS}')
    print(f'[INFO] channels: {COMMON_CHANNELS}')

    # ---- Load stimulus features ----
    feature_dim_dict = DEFAULTS['feature_dim_dict']
    if '_' not in feature_name:
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

    feat_dict_ls = []
    feat_frames_ls = []
    for feat in feature_name_ls:
        # v2: 根据特征类型使用不同的 fs (对齐 train_models_sub_v4.py)
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

    # ---- Build model ----
    # v2: in_dim 固定为公共通道数 (57), 不随 eeg_device 变化
    in_dim = NUM_COMMON_CHANNELS

    test_story_range = list(range(test_story_range_start, test_story_range_end + 1))
    print(f'Test story range: {test_story_range}')

    sub_names = ['sub-{:02d}'.format(i) for i in range(1, 26)]
    sub_ls_all = [subject_id]

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
    model.to(device)

    # ---- Load checkpoint ----
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch", -1)
    best_top1 = checkpoint.get("top1_acc", None)
    print(f'Loaded model from epoch {best_epoch}, validation top-1: {best_top1}')

    # ---- Load test data with target eeg_device ----
    print(f'\nLoading Test Set for Subject {subject_id}...')
    print(f'Source eeg_device: {source_eeg_device}, Target eeg_device: {target_eeg_device}')
    print(f'story_range: {test_story_range}')

    test_EEG, test_idx, test_sub, test_sti, test_id2dict, _ = get_EEG_emb_feat_story_range(
        sub_ls_all, feat_dict, feat_frames_id_dict, feat_keys,
        story_range=test_story_range, eeg_device=target_eeg_device,
        seg_len=seg_len, fs=250, std_eeg='std', phase='test',
        feature_name=feature_name
    )

    if len(test_id2dict) == 0:
        print(f'[WARNING] No test data found for subject {subject_id} in story range {test_story_range}')
        test_results = {
            "subject_id": subject_id,
            "data_ratio": data_ratio,
            "best_validation_epoch": best_epoch,
            "best_validation_accuracy": best_top1.item() if isinstance(best_top1, torch.Tensor) else best_top1,
            "test_accuracy": 0.0,
            "per_subject_test_accuracy": {},
            "test_story_range": test_story_range,
            "test_type": test_type,
            "in_dim": in_dim,
            "num_common_channels": NUM_COMMON_CHANNELS,
            "common_channels": COMMON_CHANNELS,
        }
        test_results_path = op.join(checkpoint_dir, output_filename)
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=4)
        print(f'Test results saved to {test_results_path}')
        return

    test_set = KUL_dataset(test_EEG, test_idx, test_sub, test_sti, test_id2dict,
                           sub_names, use_multi_band=False)
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=16)

    print(f'Test set size: {len(test_set)}')

    # ---- Evaluate ----
    test_acc, test_acc_sub_dict = evaluate(model, test_dataloader, device, embeds)

    print(f'\nTest Results:')
    print(f'  Subject: {subject_id}')
    print(f'  Test type: {test_type}')
    print(f'  Test Accuracy: {test_acc:.4f}')
    print(f'  Per-subject Test Accuracy:')
    for sub_name, acc in test_acc_sub_dict.items():
        print(f'    {sub_name}: {acc:.4f}')

    test_results = {
        "subject_id": subject_id,
        "data_ratio": data_ratio,
        "best_validation_epoch": best_epoch,
        "best_validation_accuracy": best_top1.item() if isinstance(best_top1, torch.Tensor) else best_top1,
        "test_accuracy": test_acc,
        "per_subject_test_accuracy": test_acc_sub_dict,
        "test_story_range": test_story_range,
        "test_type": test_type,
        "in_dim": in_dim,
        "num_common_channels": NUM_COMMON_CHANNELS,
        "common_channels": COMMON_CHANNELS,
    }

    test_results_path = op.join(checkpoint_dir, output_filename)
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f'Test results saved to {test_results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--subject_id', type=int, required=True, help='Subject ID to use (1-25)')
    parser.add_argument('--data_ratio', type=float, required=True, help='Data ratio (0.2, 0.4, 0.6, 0.8, 1.0)')
    parser.add_argument('--test_type', required=True,
                        choices=['day1_to_day2', 'day2_to_day1', 'neurascan_to_brk'],
                        help='Type of cross-condition test')

    args = parser.parse_args()
    test(args)
