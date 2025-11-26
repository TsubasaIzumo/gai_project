#!/bin/bash
#SBATCH --job-name=best_practice             # 作业名称
#SBATCH --nodes=1                         # 使用1个节点
#SBATCH --gpus=1                          # 请求1块GPU
#SBATCH --time=16:00:00                   # 最长运行10分钟
#SBATCH --partition=normal                # 分区（根据你集群设置确认）
#SBATCH --account=mscaisuperpod  # ✅ 替换成你的项目账户名

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

#echo "=== DIAG ==="
#echo "which python: $(which python)"
#python -V
#echo "sys.executable (env python):"
#python - << 'PY'
#import sys
#print(sys.executable)
#PY
#echo "=== END DIAG ==="

python -c "import sys; print('RUN PY:', sys.executable)"
python -m pip show pytorch-lightning || true
python -m pip show lightning || true

python train_generator.py --config configs/generator_best_practice.yaml

# python generate_images_realizations.py \
#     --ckpt "/home/ldingaj/project/gai/diffusion-for-sources-characterisation/lightning_logs/2025-11-20-04-06-12_test_run_VI/version_0/checkpoints/best_epoch=27_step=12935.ckpt" \
#     --output "./results/generated_images" \
#     --bs 5 \
#     --timestep_respacing 250 \
#     --runs_per_sample 10