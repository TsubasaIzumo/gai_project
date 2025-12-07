"""
Generate images using pretrained diffusion model
"""
import argparse
from pathlib import Path
import copy

from tqdm import tqdm
import numpy as np

import torch

from src.trainer.generator_pretrained import PretrainedGeneratorModule
from src.utils import get_config, str2bool

def torch_to_image_numpy(tensor: torch.Tensor):
    """Convert torch tensor to numpy image"""
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0) for i in range(tensor.shape[0])]
    return im_np


def load_config_from_checkpoint(checkpoint_path: str, base_config: dict) -> dict:
    """
    Load configuration from checkpoint if available.
    Allows command-line arguments to override specific config items.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Try to read hyperparameters from checkpoint
        if 'hyper_parameters' in checkpoint:
            ckpt_config = checkpoint['hyper_parameters'].get('config', None)
            if ckpt_config:
                print("✓ Loaded configuration from checkpoint")
                # Use checkpoint config as base
                merged_config = copy.deepcopy(ckpt_config)
                # But keep some runtime parameters
                merged_config['batch_size'] = base_config.get('batch_size', merged_config.get('batch_size', 50))
                return merged_config
        
        # If config not found, use provided config file
        print("⚠ Configuration not found in checkpoint, using provided config file")
        return base_config
    except Exception as e:
        print(f"⚠ Unable to read config from checkpoint: {e}")
        print("Using provided config file")
        return base_config


def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    config_path = args.config
    config = get_config(config_path)
    
    path_checkpoint = args.ckpt
    print(f"Checkpoint path: {path_checkpoint}")
    
    # Try to load config from checkpoint
    if args.use_ckpt_config:
        config = load_config_from_checkpoint(path_checkpoint, config)
    
    # Command-line arguments override config
    if args.latent_enabled is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['enabled'] = args.latent_enabled
        print(f"Overriding latent.enabled: {args.latent_enabled}")
    
    if args.latent_dim is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['dim'] = args.latent_dim
        print(f"Overriding latent.dim: {args.latent_dim}")
    
    if args.latent_project_channels is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['project_channels'] = args.latent_project_channels
        print(f"Overriding latent.project_channels: {args.latent_project_channels}")
    
    # Batch size
    config['batch_size'] = args.bs
    
    runs_per_sample = args.runs_per_sample
    num_latent_samples = args.num_latent_samples
    
    # Create output directory
    output_path = Path(args.output + f"_power{config['dataset']['power']}")
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_path}")
    
    # Load model
    print("\nLoading pretrained model...")
    try:
        module = PretrainedGeneratorModule.load_from_checkpoint(
            path_checkpoint,
            strict=False,
            config=config,
            use_fp16=config.get('fp16', False),
        )
        print("✓ Model loaded successfully")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n✗ Error: Model parameter shape mismatch")
            print(f"This usually happens when checkpoint config doesn't match current config file")
            print(f"\nCurrent config:")
            print(f"  - latent.enabled: {config.get('latent', {}).get('enabled', False)}")
            print(f"  - latent.dim: {config.get('latent', {}).get('dim', 'N/A')}")
            print(f"  - latent.project_channels: {config.get('latent', {}).get('project_channels', 'N/A')}")
            print(f"\nSuggestions:")
            print(f"1. Use --use_ckpt_config to auto-load config from checkpoint")
            print(f"2. Or manually set correct --latent_enabled, --latent_dim, --latent_project_channels")
            raise
        else:
            raise
    
    module.eval()
    module.to(device)
    
    # Get dataloader
    print(f"\nLoading data...")
    data_loader = module.get_dataloader('test')
    print(f"✓ Dataset size: {len(data_loader.dataset)}")
    print(f"✓ Batch size: {args.bs}")
    
    # Configuration info
    print("\n" + "="*60)
    print("Generation Configuration")
    print("="*60)
    pretrained_cfg = config.get('pretrained', {})
    latent_cfg = config.get('latent', {})
    print(f"Pretrained model: {pretrained_cfg.get('model_id', 'N/A')}")
    latent_enabled = latent_cfg.get('enabled', False)
    print(f"Variational inference: {'Enabled' if latent_enabled else 'Disabled'}")
    if latent_enabled:
        print(f"  - Latent dimension: {latent_cfg.get('dim', 'N/A')}")
        print(f"  - Project channels: {latent_cfg.get('project_channels', 'N/A')}")
    print(f"Runs per sample: {runs_per_sample}")
    print(f"Latent samples per generation: {num_latent_samples}")
    print(f"Total generations = {len(data_loader)} batches × {args.bs} × {runs_per_sample} × {num_latent_samples}")
    print("="*60 + "\n")
    
    # Generate images
    print("Starting image generation...\n")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc='Generating')):
            dirty_noisy = batch['dirty_noisy'].to(device)
            true_images = batch['true'].to(device)
            
            # Multiple runs
            for run_idx in range(runs_per_sample):
                # Generate multiple latent samples
                generated = module(
                    dirty_noisy, 
                    num_latent_samples=num_latent_samples,
                    progress=False
                )
                
                # generated shape: [B, M, C, H, W] or [B, C, H, W]
                if generated.dim() == 5:  # Multi-sample
                    batch_size, n_samples = generated.shape[:2]
                    for b in range(batch_size):
                        for s in range(n_samples):
                            sample_idx = batch_idx * args.bs + b
                            filename = f"batch={batch_idx:04d}_sample={sample_idx:04d}_run={run_idx:02d}_latent={s:02d}.npy"
                            save_path = output_path / filename
                            np.save(save_path, generated[b, s].cpu().numpy())
                else:  # Single sample
                    batch_size = generated.shape[0]
                    for b in range(batch_size):
                        sample_idx = batch_idx * args.bs + b
                        filename = f"batch={batch_idx:04d}_sample={sample_idx:04d}_run={run_idx:02d}.npy"
                        save_path = output_path / filename
                        np.save(save_path, generated[b].cpu().numpy())
            
            # Optional: save conditional input and true images (only for first batch)
            if batch_idx == 0:
                current_batch_size = dirty_noisy.shape[0]
                for b in range(min(current_batch_size, 10)):  # Save at most 10
                    np.save(output_path / f"input_dirty_{b:04d}.npy", dirty_noisy[b].cpu().numpy())
                    np.save(output_path / f"input_true_{b:04d}.npy", true_images[b].cpu().numpy())
    
    print(f"\n✓ Generation completed!")
    print(f"Images saved to: {output_path}")
    
    # Statistics
    generated_files = list(output_path.glob("batch=*.npy"))
    print(f"✓ Generated {len(generated_files)} files in total")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images using pretrained diffusion model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default='./results/generated_pretrained',
                        help='Output directory')
    parser.add_argument('--bs', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--runs_per_sample', type=int, default=1,
                        help='Number of runs per sample')
    parser.add_argument('--num_latent_samples', type=int, default=1,
                        help='Number of latent samples per generation')
    parser.add_argument('--use_ckpt_config', action='store_true',
                        help='Load config from checkpoint (recommended)')
    parser.add_argument('--latent_enabled', type=lambda x: None if x is None else str2bool(x),
                        default=None,
                        help='Override latent.enabled config (true/false)')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Override latent.dim config')
    parser.add_argument('--latent_project_channels', type=int, default=None,
                        help='Override latent.project_channels config')
    
    args = parser.parse_args()
    main(args)

