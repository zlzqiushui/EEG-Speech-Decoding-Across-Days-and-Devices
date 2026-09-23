"""
finetune/data_all_v2.py — 公共通道版本

与 data_all_v1.py 的差异:
  - 始终使用 day1/day2/brk 三设备的公共通道 (COMMON_CHANNELS, 57个)。
  - 移除 n_channels / target_channels 参数, in_dim 固定为 57。
  - 无论 eeg_device 是什么, 所有 EEG 数据按 COMMON_CHANNELS 顺序对齐。
  - 其余逻辑 (stimulus 加载, fs 适配, 标准化等) 与 data_all_v1.py 一致。
"""

import os
from . import config as cfg
from tqdm import tqdm
import os.path as op
import numpy as np
from torch.utils.data import Dataset
import glob
from mne.filter import filter_data


# =========================================================================
# 三设备公共通道 (57 个)
# =========================================================================
COMMON_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'Pz', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'O1', 'Oz', 'O2',
]
NUM_COMMON_CHANNELS = len(COMMON_CHANNELS)  # 57


def _align_to_common_channels(eeg, current_ch_names, recording_path):
    """将 EEG 数据对齐到 COMMON_CHANNELS 顺序。"""
    if len(eeg.shape) == 2 and eeg.shape[0] == len(current_ch_names):
        eeg = eeg.T

    align_idx = []
    for ch in COMMON_CHANNELS:
        if ch in current_ch_names:
            align_idx.append(current_ch_names.index(ch))
        else:
            raise ValueError(
                f"Common channel '{ch}' not found in {recording_path}. "
                f"Available: {current_ch_names}"
            )
    return eeg[:, align_idx]


def get_stimulus_feat(feature_name='envelope', seg_len=5, fs=100, std_feat='std'):
    _FEATURE_SUBDIR = {
        'envelope': 'env_v7',
        'bert': 'bert',
        'mel': 'mel',
        'wav2vecbase9': 'wav2vecbase9',
    }
    subdir = _FEATURE_SUBDIR.get(feature_name, feature_name)
    stimuli_dir = os.path.join(cfg.processed_stimuli_dir, subdir)
    all_stimulus_files = glob.glob(os.path.join(stimuli_dir, "*.npy"))

    _FEATURE_FS = {
        'envelope': 100,
        'mel': 50,
        'bert': 50,
        'wav2vecbase9': 50,
    }
    fs = _FEATURE_FS.get(feature_name, fs)

    feat_dict = {}
    seg_len_emb = int(fs * seg_len)

    for stimulus_file in tqdm(all_stimulus_files):
        base = os.path.basename(stimulus_file)
        name_part = os.path.splitext(base)[0]
        stimulus_name = name_part.split('_')[0]

        stimulus_feat = np.load(stimulus_file)
        print(base, stimulus_feat.shape)
        stimulus_feat = stimulus_feat.astype(np.float32)

        total_len = len(stimulus_feat)
        frame_num = total_len // seg_len_emb
        stimulus_feat = stimulus_feat[:frame_num * seg_len_emb]
        stimulus_feat_frames = stimulus_feat.reshape(frame_num, seg_len_emb, -1)

        if std_feat == 'std':
            feat_std = np.std(stimulus_feat_frames, axis=1, keepdims=True)
            feat_mean = np.mean(stimulus_feat_frames, axis=1, keepdims=True)
            stimulus_feat_frames = (stimulus_feat_frames - feat_mean) / (feat_std + 1e-8)
        elif std_feat == 'rbstd':
            center = np.median(stimulus_feat_frames, axis=1, keepdims=True)
            q1 = np.percentile(stimulus_feat_frames, 25, axis=1, keepdims=True)
            q3 = np.percentile(stimulus_feat_frames, 75, axis=1, keepdims=True)
            scaler = q3 - q1
            stimulus_feat_frames = (stimulus_feat_frames - center) / (scaler + 1e-8)
            stimulus_feat_frames = np.clip(stimulus_feat_frames, -20, 20)
        else:
            raise ValueError('std_feat should be std or rbstd')

        feat_dict[stimulus_name] = stimulus_feat_frames

    feat_keys = sorted(feat_dict.keys(), key=lambda x: int(x))
    feat_frames = np.concatenate([feat_dict[k] for k in feat_keys], axis=0)
    feat_frames_id = np.arange(feat_frames.shape[0])
    feat_frames_num = [feat_dict[k].shape[0] for k in feat_keys]
    feat_frames_id_split = np.split(feat_frames_id, np.cumsum(feat_frames_num)[0:-1])
    feat_frames_id_dict = {k: v for k, v in zip(feat_keys, feat_frames_id_split)}

    return feat_dict, feat_frames_id_dict, feat_keys, feat_frames


# =========================================================================
# 主数据加载函数 —— 始终使用公共通道对齐
# =========================================================================

def get_EEG_emb_feat_story_range(sub_ls, feat_dict, feat_frames_id_dict, feat_keys,
                                 story_range=None, seg_len=5, fs=250, std_eeg='std',
                                 phase='train', eeg_device='both',
                                 feature_name='envelope'):
    """加载 EEG 数据并按公共通道对齐。eeg_device 仅影响故事范围, 不影响通道维度。"""
    num_channels = NUM_COMMON_CHANNELS
    seg_len_emb = int(fs * seg_len)
    EEG_dict = {}
    EEG_feat_index_dict = {}
    Sub_id_dict = {}
    Stimulus_index_dict = {}
    all_subjects = [os.path.join(cfg.processed_eeg_dir, 'sub-{:02d}'.format(i)) for i in sub_ls]
    train_stimulus_filename = []

    for subject_path in tqdm(all_subjects, desc=f'Processing {phase} data (common ch)'):
        subject = os.path.basename(subject_path)

        if story_range is not None:
            all_recordings = []
            for story_num in story_range:
                story_file = os.path.join(subject_path, f"story_{story_num}.npz")
                if os.path.exists(story_file):
                    all_recordings.append(story_file)
        else:
            all_recordings = glob.glob(os.path.join(subject_path, "story_*.npz"))

        EEG_sub = []
        EEG_feat_index_sub = []
        stimulus_index_sub = []

        for recording in all_recordings:
            eeg_data = np.load(recording, allow_pickle=True)

            if 'eeg_data' in eeg_data:
                eeg = eeg_data['eeg_data']
            elif 'data' in eeg_data:
                eeg = eeg_data['data']

            current_ch_names = list(eeg_data['ch_names'])
            eeg = eeg.astype(np.float32)

            # v2: 始终按 COMMON_CHANNELS 对齐, 不区分 eeg_device
            eeg = _align_to_common_channels(eeg, current_ch_names, recording)

            base_name = os.path.basename(recording)
            stimulus_filename = base_name.split('.')[0].split('_')[1]

            if stimulus_filename == 'audiobook_1' and phase == 'train':
                continue
            if phase == 'train':
                train_stimulus_filename.append(stimulus_filename)

            stimulus_feat = feat_dict[stimulus_filename]
            stimulus_total_len = stimulus_feat.shape[0] * stimulus_feat.shape[1]
            print(f"Stimulus {stimulus_filename} has total length {stimulus_total_len} and EEG has length {len(eeg)}")
            if feature_name != 'envelope':
                stimulus_total_len = int(stimulus_total_len * 5)
            else:
                stimulus_total_len = int(stimulus_total_len * 2.5)

            if len(eeg) > stimulus_total_len:
                eeg = eeg[:stimulus_total_len]
            elif len(eeg) < stimulus_total_len:
                eeg = np.concatenate([eeg, np.zeros((stimulus_total_len - len(eeg), num_channels))], axis=0)

            n_frames = stimulus_feat.shape[0]
            eeg_frames = eeg.reshape(n_frames, seg_len_emb, num_channels)
            print(f"Reshaped EEG to {eeg_frames.shape} for stimulus {stimulus_filename}")

            EEG_sub.append(eeg_frames)
            EEG_feat_index_sub.append(feat_frames_id_dict[stimulus_filename])
            stimulus_index_sub.append(feat_keys.index(stimulus_filename) * np.ones(len(eeg_frames), dtype=np.int64))

        if len(EEG_sub) == 0:
            print(f"Warning: No EEG data for subject {subject}")
            continue

        EEG_sub = np.concatenate(EEG_sub, axis=0)

        if std_eeg == 'std':
            EEG_sub_std = np.std(EEG_sub, axis=1, keepdims=True)
            EEG_sub_mean = np.mean(EEG_sub, axis=1, keepdims=True)
            EEG_sub = (EEG_sub - EEG_sub_mean) / (EEG_sub_std + 1e-8)
        elif std_eeg == 'rbstd':
            center = np.median(EEG_sub, axis=1, keepdims=True)
            q1 = np.percentile(EEG_sub, 25, axis=1, keepdims=True)
            q3 = np.percentile(EEG_sub, 75, axis=1, keepdims=True)
            scaler = q3 - q1
            EEG_sub = (EEG_sub - center) / (scaler + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            raise ValueError('std_eeg should be std or rbstd')

        EEG_dict[subject] = EEG_sub.astype(np.float32)
        EEG_feat_index_dict[subject] = np.concatenate(EEG_feat_index_sub, axis=0)
        Sub_id_dict[subject] = np.ones(len(EEG_feat_index_dict[subject]), dtype=np.int64) * (int(subject.split('-')[-1]) - 1)
        Stimulus_index_dict[subject] = np.concatenate(stimulus_index_sub, axis=0)

    if len(EEG_dict) > 0:
        sub_frames_id = [np.arange(len(EEG_dict[k])) for k in EEG_dict.keys()]
        sub_frames_subid = [v for k, v in Sub_id_dict.items()]
        sub_frames_id = np.concatenate(sub_frames_id, axis=0)
        sub_frames_subid = np.concatenate(sub_frames_subid, axis=0)
        id2dict = np.concatenate([sub_frames_subid[:, None], sub_frames_id[:, None]], axis=1)
    else:
        id2dict = np.array([])

    return EEG_dict, EEG_feat_index_dict, Sub_id_dict, Stimulus_index_dict, id2dict, train_stimulus_filename


# =========================================================================
# 以下函数沿用 data_all_v1.py 的其余逻辑, 通道部分替换为公共通道对齐
# =========================================================================

def get_EEG_emb_feat_test_story_range(sub_ls, feat_dict, feat_frames_id_dict, feat_keys,
                                      story_range=None, train_stimulus_filename=None,
                                      seg_len=5, fs=250, std_eeg='std',
                                      feature_name='envelope'):
    num_channels = NUM_COMMON_CHANNELS
    seg_len_emb = int(fs * seg_len)
    EEG_dict = {}
    EEG_feat_index_dict = {}
    Sub_id_dict = {}
    Stimulus_index_dict = {}
    all_subjects = [op.join(cfg.processed_eeg_dir, 'sub-{:02d}'.format(i)) for i in sub_ls]

    for subject_path in tqdm(all_subjects, desc='Processing test data (common ch)'):
        subject = os.path.basename(subject_path)

        if story_range is not None:
            all_recordings = []
            for story_num in story_range:
                story_file = os.path.join(subject_path, f"story_{story_num}.npz")
                if os.path.exists(story_file):
                    all_recordings.append(story_file)
        else:
            all_recordings = glob.glob(os.path.join(subject_path, "story_*.npz"))

        EEG_sub = []
        EEG_feat_index_sub = []
        stimulus_index_sub = []

        for recording in all_recordings:
            eeg_data = np.load(recording)
            if 'data' in eeg_data:
                eeg = eeg_data['data']
            elif 'eeg' in eeg_data:
                eeg = eeg_data['eeg']
            else:
                eeg = eeg_data[list(eeg_data.keys())[0]]
            eeg = eeg.astype(np.float32)

            current_ch_names = list(eeg_data['ch_names'])
            eeg = _align_to_common_channels(eeg, current_ch_names, recording)

            base_name = os.path.basename(recording)
            stimulus_filename = base_name.split('.')[0].split('_')[1]

            if train_stimulus_filename is not None and stimulus_filename in train_stimulus_filename:
                continue

            stimulus_feat = feat_dict[stimulus_filename]
            stimulus_total_len = stimulus_feat.shape[0] * stimulus_feat.shape[1]
            print(f"Stimulus {stimulus_filename} has total length {stimulus_total_len} and EEG has length {len(eeg)}")
            if feature_name != 'envelope':
                stimulus_total_len = int(stimulus_total_len * 5)
            else:
                stimulus_total_len = int(stimulus_total_len * 2.5)

            if len(eeg) > stimulus_total_len:
                eeg = eeg[:stimulus_total_len]
            elif len(eeg) < stimulus_total_len:
                eeg = np.concatenate([eeg, np.zeros((stimulus_total_len - len(eeg), num_channels))], axis=0)

            eeg_frames = eeg.reshape(stimulus_feat.shape[0], seg_len_emb, -1)
            EEG_sub.append(eeg_frames)
            EEG_feat_index_sub.append(feat_frames_id_dict[stimulus_filename])
            stimulus_index_sub.append(feat_keys.index(stimulus_filename) * np.ones(len(eeg_frames), dtype=np.int64))

        if len(EEG_sub) == 0:
            continue

        EEG_sub = np.concatenate(EEG_sub, axis=0)

        if std_eeg == 'std':
            EEG_sub_std = np.std(EEG_sub, axis=1, keepdims=True)
            EEG_sub_mean = np.mean(EEG_sub, axis=1, keepdims=True)
            EEG_sub = (EEG_sub - EEG_sub_mean) / (EEG_sub_std + 1e-8)
        elif std_eeg == 'rbstd':
            center = np.median(EEG_sub, axis=1, keepdims=True)
            q1 = np.percentile(EEG_sub, 25, axis=1, keepdims=True)
            q3 = np.percentile(EEG_sub, 75, axis=1, keepdims=True)
            scaler = q3 - q1
            scaler = scaler.astype(np.float32)
            EEG_sub = (EEG_sub - center) / (scaler + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            raise ValueError('std_eeg should be std or rbstd')

        EEG_dict[subject] = EEG_sub.astype(np.float32)
        EEG_feat_index_dict[subject] = np.concatenate(EEG_feat_index_sub, axis=0)
        Sub_id_dict[subject] = np.ones(len(EEG_feat_index_dict[subject]), dtype=np.int64) * (int(subject.split('-')[-1]) - 1)
        Stimulus_index_dict[subject] = np.concatenate(stimulus_index_sub, axis=0)

    sub_frames_id = [np.arange(len(EEG_dict[k])) for k in EEG_dict.keys()]
    sub_frames_subid = [v for k, v in Sub_id_dict.items()]
    sub_frames_id = np.concatenate(sub_frames_id, axis=0)
    sub_frames_subid = np.concatenate(sub_frames_subid, axis=0)
    id2dict = np.concatenate([sub_frames_subid[:, None], sub_frames_id[:, None]], axis=1)

    return EEG_dict, EEG_feat_index_dict, Sub_id_dict, Stimulus_index_dict, id2dict


def get_EEG_emb_feat_all(sub_ls, feat_dict, feat_frames_id_dict, feat_keys,
                         seg_len=5, fs=250, std_eeg='std',
                         feature_name='envelope'):
    num_channels = NUM_COMMON_CHANNELS
    seg_len_emb = int(fs * seg_len)
    EEG_dict = {}
    EEG_feat_index_dict = {}
    Sub_id_dict = {}
    Stimulus_index_dict = {}
    all_subjects = [os.path.join(cfg.processed_eeg_dir, 'sub-{:02d}'.format(i)) for i in sub_ls]
    sub_segment_counts = []

    for subject_path in tqdm(all_subjects, desc='Processing all subjects (common ch)'):
        subject = os.path.basename(subject_path)
        all_recordings = glob.glob(os.path.join(subject_path, "story_*.npz"))

        if not all_recordings:
            print(f"Warning: No recordings for {subject}")
            sub_segment_counts.append(0)
            continue

        def get_story_number(path):
            basename = os.path.basename(path)
            name_no_ext = os.path.splitext(basename)[0]
            parts = name_no_ext.split('_')
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
            return 0

        all_recordings.sort(key=get_story_number)
        print(all_recordings)

        EEG_sub = []
        EEG_feat_index_sub = []
        stimulus_index_sub = []

        for recording in all_recordings:
            eeg_data = np.load(recording)
            if 'data' in eeg_data:
                eeg = eeg_data['data']
            elif 'eeg' in eeg_data:
                eeg = eeg_data['eeg']
            else:
                eeg = eeg_data[list(eeg_data.keys())[0]]
            eeg = eeg.astype(np.float32)

            current_ch_names = list(eeg_data['ch_names'])
            eeg = _align_to_common_channels(eeg, current_ch_names, recording)

            base_name = os.path.basename(recording)
            stimulus_filename = base_name.split('.')[0].split('_')[1]

            stimulus_feat = feat_dict[stimulus_filename]
            stimulus_total_len = stimulus_feat.shape[0] * stimulus_feat.shape[1]
            if feature_name != 'envelope':
                stimulus_total_len = int(stimulus_total_len * 5)
            else:
                stimulus_total_len = int(stimulus_total_len * 2.5)

            if len(eeg) > stimulus_total_len:
                eeg = eeg[:stimulus_total_len]
            elif len(eeg) < stimulus_total_len:
                eeg = np.concatenate([eeg, np.zeros((stimulus_total_len - len(eeg), num_channels))], axis=0)

            n_frames = stimulus_feat.shape[0]
            eeg_frames = eeg.reshape(n_frames, seg_len_emb, num_channels)
            EEG_sub.append(eeg_frames)
            EEG_feat_index_sub.append(feat_frames_id_dict[stimulus_filename])
            stimulus_index_sub.append(feat_keys.index(stimulus_filename) * np.ones(len(eeg_frames), dtype=np.int64))

        if not EEG_sub:
            sub_segment_counts.append(0)
            continue

        EEG_sub = np.concatenate(EEG_sub, axis=0)

        if std_eeg == 'std':
            EEG_sub_std = np.std(EEG_sub, axis=1, keepdims=True)
            EEG_sub_mean = np.mean(EEG_sub, axis=1, keepdims=True)
            EEG_sub = (EEG_sub - EEG_sub_mean) / (EEG_sub_std + 1e-8)
        elif std_eeg == 'rbstd':
            center = np.median(EEG_sub, axis=1, keepdims=True)
            q1 = np.percentile(EEG_sub, 25, axis=1, keepdims=True)
            q3 = np.percentile(EEG_sub, 75, axis=1, keepdims=True)
            scaler = q3 - q1
            EEG_sub = (EEG_sub - center) / (scaler + 1e-8)
            EEG_sub = np.clip(EEG_sub, -20, 20)
        else:
            raise ValueError('std_eeg should be std or rbstd')

        EEG_dict[subject] = EEG_sub.astype(np.float32)
        EEG_feat_index_dict[subject] = np.concatenate(EEG_feat_index_sub, axis=0)
        Sub_id_dict[subject] = np.ones(len(EEG_feat_index_dict[subject]), dtype=np.int64) * (int(subject.split('-')[-1]) - 1)
        Stimulus_index_dict[subject] = np.concatenate(stimulus_index_sub, axis=0)

        sub_segment_counts.append(len(EEG_sub))

    if len(EEG_dict) > 0:
        sub_frames_id = [np.arange(len(EEG_dict[k])) for k in EEG_dict.keys()]
        sub_frames_subid = [v for k, v in Sub_id_dict.items()]
        sub_frames_id = np.concatenate(sub_frames_id, axis=0)
        sub_frames_subid = np.concatenate(sub_frames_subid, axis=0)
        id2dict = np.concatenate([sub_frames_subid[:, None], sub_frames_id[:, None]], axis=1)
    else:
        id2dict = np.array([])

    return EEG_dict, EEG_feat_index_dict, Sub_id_dict, Stimulus_index_dict, id2dict, sub_segment_counts


class KUL_dataset(Dataset):
    def __init__(self, EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                 Stimulus_index_dict,
                 id2dict, sub_names,
                 use_multi_band=False, indices=None):

        self.band_limits = [0, 4, 8, 12, 30]
        self.use_multi_band = use_multi_band
        if use_multi_band:
            EEG_dict_new = {}
            for k, v in tqdm(EEG_dict.items(), desc='Filtering EEG'):
                eeg = EEG_dict[k]
                eeg = np.swapaxes(eeg, 1, 2)
                shape = eeg.shape
                eeg = np.reshape(eeg, (eeg.shape[0] * eeg.shape[1], -1))
                eeg = eeg.astype(np.float64)
                eeg_new_ls = []
                for i in range(len(self.band_limits) - 1):
                    low, high = self.band_limits[i], self.band_limits[i + 1]
                    eeg_i = filter_data(eeg, 64, low, high,
                                        l_trans_bandwidth=1, h_trans_bandwidth=1,
                                        verbose='warning', filter_length=256, n_jobs=16)
                    eeg_i = np.reshape(eeg_i, shape)
                    eeg_i = np.swapaxes(eeg_i, 1, 2)
                    eeg_i = eeg_i.astype(np.float32)
                    eeg_new_ls.append(eeg_i)
                eeg_new = np.concatenate(eeg_new_ls, axis=2)
                eeg_new = (eeg_new - np.mean(eeg_new, axis=1, keepdims=True)) / (np.std(eeg_new, axis=1, keepdims=True) + 1e-8)
                EEG_dict_new[k] = eeg_new
            self.EEG_dict = EEG_dict_new
        else:
            self.EEG_dict = EEG_dict

        self.EEG_feat_index_dict = EEG_feat_index_dict
        self.Sub_id_dict = Sub_id_dict
        self.Stimulus_index_dict = Stimulus_index_dict
        if indices is not None:
            self.id2dict = id2dict[indices]
        else:
            self.id2dict = id2dict
        self.sub_names = sub_names
        self.data_len = len(self.id2dict)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        t = self.id2dict[index]
        sub_idx, frame_idx = int(t[0]), int(t[1])
        sub_name = self.sub_names[sub_idx]
        emb_id = self.EEG_feat_index_dict[sub_name][frame_idx]
        eeg = self.EEG_dict[sub_name][frame_idx]
        sub_id = self.Sub_id_dict[sub_name][frame_idx]
        sti_id = self.Stimulus_index_dict[sub_name][frame_idx]
        return eeg, emb_id, sub_id, sti_id
