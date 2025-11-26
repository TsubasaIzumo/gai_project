import argparse
from pathlib import Path
import copy

from tqdm import tqdm
import numpy as np

import torch

from src.trainer import GeneratorModule
from src.utils import get_config


def torch_to_image_numpy(tensor: torch.Tensor):
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0) for i in range(tensor.shape[0])]
    return im_np


def load_config_from_checkpoint(checkpoint_path: str, base_config: dict) -> dict:
    """
    从checkpoint中加载配置信息，如果存在则使用checkpoint中的配置。
    允许通过命令行参数覆盖特定配置项。
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 尝试从checkpoint中读取hyperparameters
        if 'hyper_parameters' in checkpoint:
            ckpt_config = checkpoint['hyper_parameters'].get('config', None)
            if ckpt_config:
                print("从checkpoint中读取配置信息")
                # 使用checkpoint中的配置作为基础
                merged_config = copy.deepcopy(ckpt_config)
                # 但保留一些运行时参数
                merged_config['batch_size'] = base_config.get('batch_size', merged_config.get('batch_size', 50))
                return merged_config
        
        # 如果没有找到配置，尝试从state_dict推断
        print("警告: checkpoint中未找到配置信息，使用提供的配置文件")
        return base_config
    except Exception as e:
        print(f"警告: 无法从checkpoint读取配置: {e}")
        print("使用提供的配置文件")
        return base_config


def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = args.config
    config = get_config(config_path)
    
    path_checkpoint = args.ckpt
    
    # 尝试从checkpoint加载配置（如果启用）
    if args.use_ckpt_config:
        config = load_config_from_checkpoint(path_checkpoint, config)
    
    # 允许通过命令行参数覆盖latent配置（在从checkpoint加载后仍然生效）
    if args.latent_enabled is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['enabled'] = args.latent_enabled
    
    if args.latent_dim is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['dim'] = args.latent_dim
    
    if args.latent_project_channels is not None:
        if 'latent' not in config:
            config['latent'] = {}
        config['latent']['project_channels'] = args.latent_project_channels
    
    # 命令行参数始终覆盖配置文件中的batch_size
    config['batch_size'] = args.batch_size

    runs_per_sample = args.runs_per_sample
    
    output_path = Path(args.output+f"_power{config['dataset']['power']}")
    output_path.mkdir(exist_ok=True, parents=True)

    # 使用更宽松的参数加载策略
    try:
        module = GeneratorModule.load_from_checkpoint(
            checkpoint_path=path_checkpoint, 
            strict=False,
            config=config, 
            use_fp16=config.get('fp16', False),
            timestep_respacing=str(args.timestep_respacing)
        )
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"\n错误: 模型参数形状不匹配")
            print(f"这通常是因为checkpoint的配置与当前配置文件不匹配")
            print(f"\n当前配置:")
            print(f"  - latent.enabled: {config.get('latent', {}).get('enabled', False)}")
            print(f"  - latent.dim: {config.get('latent', {}).get('dim', 'N/A')}")
            print(f"  - latent.project_channels: {config.get('latent', {}).get('project_channels', 'N/A')}")
            print(f"  - dataset.n_channels: {config.get('dataset', {}).get('n_channels', 'N/A')}")
            print(f"\n建议解决方案:")
            print(f"  1. 使用 --use_ckpt_config 从checkpoint读取配置")
            print(f"  2. 或使用以下参数覆盖latent配置:")
            print(f"     --latent_enabled True/False")
            print(f"     --latent_dim <dim>")
            print(f"     --latent_project_channels <channels>")
            print(f"\n详细错误信息:\n{e}")
            raise
        else:
            raise
    
    module.eval()

    # Move all model components to device
    diffusion = module.diffusion
    model_ema = module.model_ema.to(device)
    
    # Move latent components to device if latent is enabled
    if module.latent_enabled:
        module.prior_net = module.prior_net.to(device)
        module.encoder_net = module.encoder_net.to(device)
        module.latent_projector = module.latent_projector.to(device)

    def model_fn(x_t, ts, **kwargs):
        """
        Model function for classifier-free guidance.
        Handles both conditional and unconditional generation.
        
        Note: x_t is already concatenated [x, cond, latent] from p_mean_variance.
        For classifier-free guidance, we take the first half of x (noise), duplicate it,
        and apply it with conditional input (dirty_noisy) and unconditional input (zeros).
        """
        # x_t is already concatenated as [x, cond, latent] from p_mean_variance
        # Split it back into components
        n_channels = module.n_channels
        
        # Split the concatenated input
        x = x_t[:, :n_channels]  # noise, shape: [batch_size * 2, n_channels, H, W]
        cond = x_t[:, n_channels:2*n_channels]  # conditional input, shape: [batch_size * 2, n_channels, H, W]
        if module.latent_enabled:
            # Correctly calculate latent channels
            latent_channels = module.latent_channels
            latent = x_t[:, 2*n_channels:2*n_channels+latent_channels]  # latent, shape: [batch_size * 2, latent_channels, H, W]
        else:
            latent = None
        
        # For classifier-free guidance: use first half of x, duplicate it
        half_batch = len(x_t) // 2
        x_half = x[:half_batch]  # Take first half: [batch_size, n_channels, H, W]
        
        # Conditional branch: first half of x + first half of cond (dirty_noisy)
        cond_cond = cond[:half_batch]  # This is dirty_noisy
        inputs_cond = [x_half, cond_cond]
        if latent is not None:
            latent_cond = latent[:half_batch]
            inputs_cond.append(latent_cond)
        combined_cond = torch.cat(inputs_cond, dim=1)
        
        # Unconditional branch: first half of x + zero condition
        cond_uncond = torch.zeros_like(cond_cond)  # Zero condition for unconditional
        inputs_uncond = [x_half, cond_uncond]
        if latent is not None:
            latent_uncond = latent[:half_batch]  # Use same latent
            inputs_uncond.append(latent_uncond)
        combined_uncond = torch.cat(inputs_uncond, dim=1)
        
        # Combine both branches: [conditional_batch, unconditional_batch]
        combined = torch.cat([combined_cond, combined_uncond], dim=0)
        
        # Call model
        model_out = model_ema(combined, ts, **kwargs)
        
        # Split output: first n_channels are noise prediction, rest are additional outputs
        eps, rest = model_out[:, :n_channels], model_out[:, n_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        
        # Apply classifier-free guidance
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        
        # Duplicate the guided eps to match the expected output shape
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # Generate images for validation and test sets
    dls = [module.val_dataloader(), module.test_dataloader()]
    names = ['val', 'test']

    for i, dl in enumerate(dls):
        print(f"Processing {names[i]} dataset...")
        for batch_idx, batch in tqdm(enumerate(dl)):
            print(f"Processing batch {batch_idx} for {names[i]} dataset")

            generated_images = []
            dirty_noisy_list = []
            sky_indexes_list = []

            dirty_noisy = batch["dirty_noisy"].to(device)
            filenames = batch["filename"]
            
            # Get actual batch size from data
            actual_batch_size = dirty_noisy.shape[0]

            for _ in tqdm(range(runs_per_sample)):
                with torch.no_grad():
                    zero_label_noise = torch.zeros_like(dirty_noisy, device=device)
                    dirty_noisy_combined = torch.cat([dirty_noisy, zero_label_noise], dim=0)
                    
                    # Calculate shape based on actual batch size
                    # For classifier-free guidance, we need batch_size * 2
                    shape = (actual_batch_size * 2, module.n_channels, module.size_image, module.size_image)
                    
                    # Prepare model_kwargs for latent if enabled
                    model_kwargs = {}
                    if module.latent_enabled:
                        mu_p, log_var_p = module.prior_net(dirty_noisy)
                        std_p = torch.exp(0.5 * log_var_p)
                        z = mu_p + std_p * torch.randn_like(std_p)
                        
                        # Debug: Print z shape and statistics (only first time)
                        if batch_idx == 0 and _ == 0:
                            print(f"\n=== Latent Variable Debug Info ===")
                            print(f"z shape: {z.shape}")
                            print(f"z dtype: {z.dtype}")
                            print(f"z min: {z.min().item():.4f}, max: {z.max().item():.4f}, mean: {z.mean().item():.4f}, std: {z.std().item():.4f}")
                            print(f"latent_dim: {module.latent_dim}")
                            print(f"latent_channels (project_channels): {module.latent_channels}")
                            print(f"dirty_noisy shape: {dirty_noisy.shape}")
                            
                        spatial_size = dirty_noisy.shape[-2:]
                        latent_map = module.latent_projector(z, spatial_size=spatial_size)
                        
                        # Debug: Print latent_map shape (only first time)
                        if batch_idx == 0 and _ == 0:
                            print(f"latent_map shape: {latent_map.shape}")
                            print(f"latent_map min: {latent_map.min().item():.4f}, max: {latent_map.max().item():.4f}")
                            print(f"Expected input channels: {module.n_channels * 2 + module.latent_channels}")
                            print(f"===================================\n")
                        
                        # Duplicate latent_map to match dirty_noisy_combined's batch dimension
                        # This is needed for classifier-free guidance
                        latent_map = torch.cat([latent_map, latent_map], dim=0)
                        model_kwargs["latent"] = latent_map
                    
                    im_out = diffusion.p_sample_loop(
                        model_fn,
                        cond=dirty_noisy_combined,
                        shape=shape,
                        device=device,
                        clip_denoised=module.clip_denoised,
                        progress=args.progress,
                        cond_fn=None,
                        model_kwargs=model_kwargs,
                    )[:actual_batch_size]
                    
                    # Debug: Print generated image statistics (only first time)
                    if batch_idx == 0 and _ == 0:
                        print(f"\n=== Generated Image Debug Info ===")
                        print(f"Generated image shape: {im_out.shape}")
                        print(f"Generated image min: {im_out.min().item():.4f}, max: {im_out.max().item():.4f}")
                        print(f"Generated image mean: {im_out.mean().item():.4f}, std: {im_out.std().item():.4f}")
                        print(f"Dirty noisy min: {dirty_noisy.min().item():.4f}, max: {dirty_noisy.max().item():.4f}")
                        print(f"Dirty noisy mean: {dirty_noisy.mean().item():.4f}, std: {dirty_noisy.std().item():.4f}")
                        print(f"Guidance scale: {args.guidance_scale}")
                        print(f"Timestep respacing: {args.timestep_respacing}")
                        print(f"====================================\n")

                im_out = torch_to_image_numpy(im_out)
                dirty_noisy_ = torch_to_image_numpy(dirty_noisy)

                generated_images.extend(im_out)
                dirty_noisy_list.extend(dirty_noisy_)
                sky_indexes_list.extend(filenames)

            generated_images = np.array(generated_images)

            np.save(output_path / f'batch={batch_idx}_{names[i]}_generated_images.npy', generated_images)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_dirty_noisy.npy', dirty_noisy_list)
            np.save(output_path / f'batch={batch_idx}_{names[i]}_sky_indexes.npy', sky_indexes_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='生成图像实现，支持从checkpoint读取配置或通过命令行参数覆盖配置'
    )
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/generator.yaml',
                        help='配置文件路径')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出文件夹')
    parser.add_argument('--batch_size', '--bs', '-b', type=int,
                        default=50,
                        help='生成样本时使用的批次大小')
    parser.add_argument('--guidance_scale', '-s', type=float,
                        default=3.0,
                        help='分类器自由引导的引导尺度')
    parser.add_argument('--timestep_respacing', '-t',
                        type=int, default=250,
                        help='时间步重采样数量')
    parser.add_argument('--runs_per_sample',
                        type=int, default=20,
                        help='每个样本的运行次数')
    parser.add_argument('--progress', action='store_true',
                        help='显示生成进度条')
    
    # Latent相关参数，允许覆盖配置
    parser.add_argument('--latent_enabled', type=lambda x: x.lower() == 'true',
                        default=None,
                        help='是否启用latent变量 (True/False)')
    parser.add_argument('--latent_dim', type=int,
                        default=None,
                        help='Latent维度，覆盖配置文件中的latent.dim')
    parser.add_argument('--latent_project_channels', type=int,
                        default=None,
                        help='Latent投影通道数，覆盖配置文件中的latent.project_channels')
    
    # 配置加载选项
    parser.add_argument('--use_ckpt_config', action='store_true',
                        help='优先使用checkpoint中保存的配置（如果存在）')
    
    args = parser.parse_args()
    main(args)

