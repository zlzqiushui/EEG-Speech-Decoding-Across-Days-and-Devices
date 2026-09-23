"""
finetune v2 配置 —— 复用项目级 config.py 的数据路径，修正模型路径指向 result_v3。
"""
import os.path as op
import os
import sys

# 指向父项目目录
project_dir = op.dirname(op.dirname(op.realpath(__file__)))
sys.path.insert(0, project_dir)

# 复用父项目的数据路径 (项目级 config.py)
import config as proj_cfg

processed_eeg_dir = proj_cfg.processed_eeg_dir
processed_stimuli_dir = proj_cfg.processed_stimuli_dir
wav2vec_model_name = proj_cfg.wav2vec_model_name
wav2vec_model_path = proj_cfg.wav2vec_model_path
gpt_model_name = proj_cfg.gpt_model_name
gpt_model_path = proj_cfg.gpt_model_path

# finetune v2 专用路径
finetune_dir = op.join(project_dir, 'finetune')
# 预训练模型来源: 指向 result_v3 (train_models_sub_v3.py 的输出)
result_v1_dir = op.join(project_dir, 'result_v3')  # 名字保留 v1 以兼容现有代码
# finetune 结果保存到新目录
finetune_result_dir = op.join(project_dir, 'finetune_result_v2')
