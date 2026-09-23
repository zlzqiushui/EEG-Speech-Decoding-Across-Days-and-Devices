#!/bin/bash
# =============================================================================
# SLURM 作业脚本 — Scaling Law 实验 (单任务串行版)
# 用法:
#     sbatch train_sub_v2.sh
#
#     squeue -u $USER                                  # 查看自己作业状态
#     scancel <JOBID>                                  # 取消作业
# =============================================================================

# ------------------ SLURM 资源申请 ------------------
#SBATCH --job-name=pretrain            # 作业名称
#SBATCH --output=logs/pretrain_%j.out               # 标准输出 (%j=JobID)
#SBATCH --error=logs/pretrain_%j.err                # 错误输出
#SBATCH --partition=GPUA800                          # GPU 分区 (A800, 最长 25 天)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7                          # 申请 1 块 GPU
#SBATCH --time=7-00:00:00                            # 最长运行时间 7 天
#SBATCH --qos=normal                                # QoS 等级

# =============================================================================
# 环境初始化
# =============================================================================
echo "========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job started"
echo "  Job ID:          ${SLURM_JOB_ID}"
echo "  Host:            $(hostname)"
echo "  Node:            ${SLURMD_NODENAME}"
echo "========================================================================="

# 加载模块
source /gpfs/share/software/module/tools/modules/init/profile.sh 2>/dev/null
module add cuda/12.8

# 用你自己的 miniconda 初始化并激活 AAD 环境
source /gpfs/share/home/2301111611/miniconda3/etc/profile.d/conda.sh
conda activate AAD

# 验证 GPU 可见性
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi failed"
echo ""

# 进入项目目录
PROJECT_DIR="/gpfs/share/home/2301111611/EEG-Stimulus-Match-Mismatch"
cd "${PROJECT_DIR}" || { echo "ERROR: Cannot cd to ${PROJECT_DIR}"; exit 1; }
mkdir -p logs

# =============================================================================
# 定义要遍历的参数列表
# =============================================================================
SUBJECTS=({1..25})
RATIOS=(0.2 0.4 0.6 0.8 1.0)
EEG_DEVICES="neurascan"  # 可选: day1 / day2 / brk
# RATIOS=(0.2)

echo "Starting Scaling Law Experiments..."
echo "EEG Device ${EEG_DEVICES}"

for sub in "${SUBJECTS[@]}"; do
    for ratio in "${RATIOS[@]}"; do

        echo "================================================================"
        echo "[RUNNING] Subject ID: ${sub} | Data Ratio: ${ratio}"
        echo "================================================================"

        # 三条 GPU 任务并行跑 (day1 / day2 / brk 共享 cuda:0)
        # 注意: 单卡同时跑三个进程可能 OOM，若显存不足请改回串行或减 batch_size
        echo "  Launching in parallel..."

        python train_models_sub_v4.py \
            --subject_id ${sub} \
            --data_ratio ${ratio} \
            --eeg_device ${EEG_DEVICES} \
            --device cuda:0 \
            --feature_name mel_10 \
            --seed 42
        # pid_day1=$!

        # python train_models_sub_v2.py \
        #     --subject_id ${sub} \
        #     --data_ratio ${ratio} \
        #     --eeg_device neurascan \
        #     --device cuda:0 &
        # pid_day2=$!

        # python train_models_sub_v2.py \
        #     --subject_id ${sub} \
        #     --data_ratio ${ratio} \
        #     --eeg_device brk \
        #     --device cuda:0 &
        # pid_brk=$!

        # # 等待三个进程全部结束，逐个检查退出状态
        # wait ${pid_day1}; rc1=$?
        # wait ${pid_day2}; rc2=$?
        # # wait ${pid_brk}; rc3=$?

        # failed=""
        # [ ${rc1} -ne 0 ] && failed="${failed} both"
        # [ ${rc2} -ne 0 ] && failed="${failed} neurascan"
        # # [ ${rc3} -ne 0 ] && failed="${failed} brk"

        # if [ -n "${failed}" ]; then
        #     echo "[ERROR] Experiment(s) failed for Subject=${sub}, Ratio=${ratio}:${failed}. Continuing to next..."
        # fi

        echo "[FINISHED] Subject ID: ${sub} | Data Ratio: ${ratio}"
        echo ""

    done
done

echo ""
echo "========================================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All Scaling Law experiments have finished."
echo "========================================================================="
