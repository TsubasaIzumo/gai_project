#!/bin/bash
#SBATCH --job-name=pretrained_ddpm         # 作业名称
#SBATCH --nodes=1                          # 使用1个节点
#SBATCH --gpus=1                           # 请求1块GPU
#SBATCH --time=00:08:00                    # 最长运行24小时
#SBATCH --partition=normal                 # 分区（根据你集群设置确认）
#SBATCH --account=mscaisuperpod            # 替换成你的项目账户名

# 预训练扩散模型训练脚本
# 使用HuggingFace预训练DDPM + 变分推断架构

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

# 设置PYTHONPATH以确保可以导入src模块
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

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
echo "Checking dependencies..."
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import sys; print('PYTHONPATH:', sys.path)"
python -m pip show pytorch-lightning || true
python -m pip show lightning || true

# 检查并安装 diffusers（必需）
echo "Checking diffusers package..."
if ! python -c "import diffusers" 2>/dev/null; then
    echo "⚠ diffusers not installed, installing now..."
    python -m pip install "diffusers>=0.21.0"
    if [ $? -ne 0 ]; then
        echo "✗ Failed to install diffusers"
        exit 1
    fi
    echo "✓ diffusers installed successfully"
else
    echo "✓ diffusers is already installed"
fi
echo ""

# 验证模块导入
echo "Verifying module imports..."
python -c "
import sys
import traceback
sys.path.insert(0, '.')

# First, try importing diffusers
print('Step 1: Checking diffusers import...')
try:
    import diffusers
    from diffusers import UNet2DModel, DDPMScheduler
    print('  ✓ diffusers imported successfully')
except Exception as e:
    print(f'  ✗ Failed to import diffusers: {e}')
    traceback.print_exc()
    sys.exit(1)

# Second, try importing the module directly
print('Step 2: Trying direct import from generator_pretrained...')
try:
    from src.trainer.generator_pretrained import PretrainedGeneratorModule
    print('  ✓ Direct import from generator_pretrained works')
except Exception as e:
    print(f'  ✗ Direct import failed: {e}')
    traceback.print_exc()
    sys.exit(1)

# Third, try importing from __init__
print('Step 3: Trying import from src.trainer...')
try:
    from src.trainer import PretrainedGeneratorModule
    if PretrainedGeneratorModule is None:
        raise ImportError('PretrainedGeneratorModule is None')
    print('  ✓ Import from src.trainer works')
    print('✓ PretrainedGeneratorModule imported successfully')
except Exception as e:
    print(f'  ✗ Import from src.trainer failed: {e}')
    traceback.print_exc()
    sys.exit(1)
" || exit 1
echo ""

# 创建必要的目录
mkdir -p checkpoints_pretrained
mkdir -p results

# 运行训练
echo "=========================================="
echo "Training Pre-trained DDPM Model"
echo "=========================================="
echo "config: configs/generator_pretrained.yaml"
echo ""

python train_pretrained.py \
    --config configs/generator_pretrained.yaml \
    --gpus 1

echo ""
echo "=========================================="
echo "Training Completed"
echo "End Time: $(date)"
echo "=========================================="

