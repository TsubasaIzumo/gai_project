#!/bin/bash
#SBATCH --job-name=gen_pretrained          # 作业名称
#SBATCH --nodes=1                          # 使用1个节点
#SBATCH --gpus=1                           # 请求1块GPU
#SBATCH --time=02:00:00                    # 最长运行2小时
#SBATCH --partition=normal                 # 分区（根据你集群设置确认）
#SBATCH --account=mscaisuperpod            # 替换成你的项目账户名

# 预训练扩散模型图像生成脚本

module purge
module load Anaconda3/2023.09-0           # 按照你系统已有的版本加载

if command -v conda >/dev/null 2>&1; then
eval "$(conda shell.bash hook)"
else
source "$(/cm/shared/apps/Anaconda3/2023.09-0/bin/conda info --base)/etc/profile.d/conda.sh" || true
source "$(/cm/shared/apps/Anaconda3/2023.09-0/bin/conda) shell.bash hook" || true
fi

conda activate astroDDPM

cd /home/ldingaj/project/gai/diffusion-for-sources-characterisation

# 检查环境
echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Python: $(which python)"
python -V
echo "=========================================="
echo ""

# 检查关键依赖
echo "检查依赖包..."
python -c "import sys; print('Python executable:', sys.executable)"
python -m pip show pytorch-lightning || true
python -m pip show diffusers || echo "警告: diffusers未安装"
echo ""

# 设置checkpoint路径（修改为你的实际路径）
CKPT_PATH="checkpoints_pretrained/pretrained-best.ckpt"

# 如果未设置，尝试查找最新的checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "未找到指定checkpoint，尝试查找最新的..."
    CKPT_PATH=$(ls -t checkpoints_pretrained/*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "错误: 未找到任何checkpoint文件"
        exit 1
    fi
fi

echo "使用checkpoint: $CKPT_PATH"
echo ""

# 创建输出目录
OUTPUT_DIR="./results/generated_pretrained_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# 运行生成
echo "=========================================="
echo "开始生成图像"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""

python generate_images_pretrained.py \
    --config "./configs/generator_pretrained.yaml" \
    --ckpt "$CKPT_PATH" \
    --output "$OUTPUT_DIR" \
    --bs 5 \
    --runs_per_sample 10 \
    --num_latent_samples 5

echo ""
echo "=========================================="
echo "生成完成"
echo "End Time: $(date)"
echo "输出保存在: $OUTPUT_DIR"
echo "=========================================="

