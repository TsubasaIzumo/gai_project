#!/bin/bash
#SBATCH --job-name=astro_test             # 作业名称
#SBATCH --nodes=1                         # 使用1个节点
#SBATCH --gpus=1                          # 请求1块GPU
#SBATCH --time=00:15:00                   # 最长运行10分钟
#SBATCH --partition=normal                # 分区（根据你集群设置确认）
#SBATCH --account=mscaisuperpod  # ✅ 替换成你的项目账户名

module purge
module load Anaconda3/2023.09-0           # 按照你系统已有的版本加载

# 激活你之前创建的环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate astroDDPM

# 切换到你的项目目录（⚠️ 替换成你的实际路径）
cd /home/ldingaj/project/gai/diffusion-for-sources-characterisation

# 启动训练脚本
python train_generator.py --config configs/generator.yaml