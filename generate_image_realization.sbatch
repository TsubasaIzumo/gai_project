#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

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

python generate_images_realizations.py \
    --config "./configs/generator_fixed.yaml" \
    --ckpt "/home/ldingaj/project/gai/diffusion-for-sources-characterisation/lightning_logs/2025-11-22-11-51-23_conditional_prior_fixed_quickfix/version_0/checkpoints/best_epoch=41_step=19403.ckpt" \
    --output "./results/generated_images" \
    --timestep_respacing 250 \
    --guidance_scale 7.0 \
    --bs 5 \
    --runs_per_sample 10 \
    --latent_enabled True \
    --latent_dim 32 \
    --latent_project_channels 4
