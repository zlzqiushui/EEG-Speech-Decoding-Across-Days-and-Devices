#!/bin/bash
# =============================================================================
# SLURM 作业脚本 — Module B v3 微调 (公共通道版本, in_dim=57)
#
# 直接调用 finetune/train_finetune_v3.py
# 使用 finetune/data_all_v2.py (公共通道对齐)
# 结果保存到 finetune_result_v3
#
# 用法:
#     sbatch finetune/train_finetune_v3.sh
#     squeue -u $USER
#     scancel <JOBID>
# =============================================================================

# ------------------ SLURM 资源申请 ------------------
#SBATCH --job-name=finetune_v3
#SBATCH --output=logs/finetune_v3_%j.out
#SBATCH --error=logs/finetune_v3_%j.err
#SBATCH --partition=GPUA800
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=7-00:00:00
#SBATCH --qos=normal

# =============================================================================
# 环境初始化
# =============================================================================
echo "========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job started"
echo "  Job ID:          ${SLURM_JOB_ID}"
echo "  Host:            $(hostname)"
echo "  Node:            ${SLURMD_NODENAME}"
echo "========================================================================="

source /gpfs/share/software/module/tools/modules/init/profile.sh 2>/dev/null
module add cuda/12.8

source /gpfs/share/home/2301111611/miniconda3/etc/profile.d/conda.sh
conda activate AAD

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi failed"
echo ""

PROJECT_DIR="/gpfs/share/home/2301111611/EEG-Stimulus-Match-Mismatch"
cd "${PROJECT_DIR}" || { echo "ERROR: Cannot cd to ${PROJECT_DIR}"; exit 1; }
mkdir -p logs

# =============================================================================
# ======================== 可配置参数 ================================
# =============================================================================

# --- 特征类型: envelope / mel / bert ---
FEATURE_NAME="mel_10"

# --- 迁移场景: day1_to_day2 / day2_to_day1 / neurascan_to_brk ---
TRANSFER="neurascan_to_brk"

# --- 策略列表: 1=Zero-shot, 2=Pretrain+Finetune, 3=Target-only, 4=Target+Historical, 5=Joint ---
STRATEGIES=(1 2 3 4 5)

# --- 被试和目标数据比例 ---
SUBJECTS=({1..25})
RATIOS=(0.2 0.4 0.6 0.8 1.0)

# --- 训练参数 ---
GPU=0
NUM_EPOCHS=150
BATCH_SIZE=32
LR=0.0002
EARLY_STOP=20
SEED=42
NUM_WORKERS=16

# =============================================================================
# 开始实验
# =============================================================================

echo "Starting Finetune v3 Experiments (common channels, in_dim=57)..."
echo "  Feature:    ${FEATURE_NAME}"
echo "  Transfer:   ${TRANSFER}"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Subjects:   ${SUBJECTS[*]}"
echo "  Ratios:     ${RATIOS[*]}"
echo "  GPU:        ${GPU}"
echo ""

for strategy in "${STRATEGIES[@]}"; do
for sub in "${SUBJECTS[@]}"; do
    for ratio in "${RATIOS[@]}"; do

        echo "================================================================"
        echo "[RUNNING] Strategy: ${strategy} | Subject: ${sub} | Ratio: ${ratio} | ${TRANSFER} | ${FEATURE_NAME}"
        echo "================================================================"

        python finetune/train_finetune_v3.py \
            --subject_id "${sub}" \
            --transfer "${TRANSFER}" \
            --strategy "${strategy}" \
            --data_ratio "${ratio}" \
            --device "cuda:${GPU}" \
            --feature_name "${FEATURE_NAME}" \
            --model_name BrainNetworkCL \
            --seg_len 5 \
            --fs 250 \
            --opt Adam \
            --lr "${LR}" \
            --num_epochs "${NUM_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --negsample_num 32 \
            --kernel_size 3 \
            --valid 1 \
            --att_out_dim 256 \
            --dropout 0.5 \
            --early_stop_num "${EARLY_STOP}" \
            --seed "${SEED}" \
            --num_workers "${NUM_WORKERS}"

        rc=$?
        if [ ${rc} -ne 0 ]; then
            echo "[ERROR] Failed: strategy=${strategy}, sub=${sub}, ratio=${ratio} (exit=${rc})"
        fi

        echo "[FINISHED] Strategy: ${strategy} | Subject: ${sub} | Ratio: ${ratio}"
        echo ""

    done
done
done

echo ""
echo "========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All v3 experiments finished."
echo "========================================================================="
