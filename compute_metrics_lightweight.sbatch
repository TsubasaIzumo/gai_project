#!/bin/bash
#SBATCH --job-name=compute_metrics      # 作业名称
#SBATCH --nodes=1                        # 使用1个节点
#SBATCH --gpus=1                         # 不需要GPU（计算指标不需要GPU）
#SBATCH --time=02:00:00                  # 最长运行2小时
#SBATCH --partition=normal               # 分区（根据你集群设置确认）
#SBATCH --account=mscaisuperpod           # 替换成你的项目账户名

# 计算重建指标脚本
# 从生成的结果文件夹中计算重建指标

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
python -m pip show numpy pandas scikit-image || echo "警告: 某些依赖包未安装"
echo ""

# 设置参数
# 如果通过命令行参数传入，使用命令行参数；否则使用默认值
OUTPUT_DIR="${1:-./results/generated_pretrained_20251205_190211_power2}"
RUNS_PER_SAMPLE="${2:-1}"
MAX_SAMPLES="${3:-45}"
CONFIG_FILE="${4:-./configs/generate_lightweight.yaml}"

# 验证输出目录是否存在
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "错误: 输出目录不存在: $OUTPUT_DIR"
    echo "用法: sbatch compute_metrics_lightweight.sbatch [输出目录] [每个样本运行次数] [最大样本数] [配置文件]"
    echo "示例: sbatch compute_metrics_lightweight.sbatch ./results/generated_pretrained_20251205_190211_power2 1 45 ./configs/generate_lightweight.yaml"
    exit 1
fi

echo "=========================================="
echo "计算重建指标配置"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo "配置文件: $CONFIG_FILE"
echo "每个样本运行次数: $RUNS_PER_SAMPLE"
echo "最大处理样本数: $MAX_SAMPLES"
echo "=========================================="
echo ""

# 检查输出目录中的文件
echo "检查输出目录中的文件..."
FILE_COUNT=$(find "$OUTPUT_DIR" -name "batch=*sample=*.npy" 2>/dev/null | wc -l)
echo "找到 $FILE_COUNT 个生成图像文件"
echo ""

# 运行计算重建指标
echo "=========================================="
echo "开始计算重建指标..."
echo "=========================================="
echo ""

python compute_reconstruction_metrics.py \
    --config "$CONFIG_FILE" \
    --runs_per_sample $RUNS_PER_SAMPLE \
    --max_samples $MAX_SAMPLES \
    --folders "$OUTPUT_DIR"

METRICS_EXIT_CODE=$?

if [ $METRICS_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "错误: 重建指标计算失败"
    echo "退出码: $METRICS_EXIT_CODE"
    echo "=========================================="
    exit $METRICS_EXIT_CODE
else
    echo ""
    echo "=========================================="
    echo "重建指标计算完成"
    echo "End Time: $(date)"
    echo "指标文件保存在: $OUTPUT_DIR/*_reconstruction_metrics.csv"
    echo "=========================================="
    
    # 列出生成的指标文件
    echo ""
    echo "生成的指标文件:"
    ls -lh "$OUTPUT_DIR"/*_reconstruction_metrics.csv 2>/dev/null || echo "未找到指标文件"
fi

echo ""
echo "=========================================="
echo "全部完成"
echo "最终结束时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

