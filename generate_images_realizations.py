import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

import torch

from src.trainer import GeneratorModule
from src.utils import get_config


def torch_to_image_numpy(tensor: torch.Tensor):
    tensor = tensor * 0.5 + 0.5
    im_np = [tensor[i].cpu().numpy().transpose(1, 2, 0) for i in range(tensor.shape[0])]
    return im_np


def main(args) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path = args.config
    config = get_config(config_path)
    config['batch_size'] = args.batch_size

    runs_per_sample = args.runs_per_sample

    path_checkpoint = args.ckpt
    output_path = Path(args.output+f"_power{config['dataset']['power']}")
    output_path.mkdir(exist_ok=True, parents=True)

    module = GeneratorModule.load_from_checkpoint(checkpoint_path=path_checkpoint, strict=False,
                                                  config=config, use_fp16=config['fp16'],
                                                  timestep_respacing=str(args.timestep_respacing))
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
            latent = x_t[:, 2*n_channels:]  # latent, shape: [batch_size * 2, latent_channels, H, W]
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
                        spatial_size = dirty_noisy.shape[-2:]
                        latent_map = module.latent_projector(z, spatial_size=spatial_size)
                        # Duplicate latent_map to match dirty_noisy_combined's batch dimension
                        # This is needed for classifier-free guidance
                        latent_map = torch.cat([latent_map, latent_map], dim=0)
                        model_kwargs["latent"] = latent_map
                    
                    im_out = diffusion.p_sample_loop(
                        model_fn,
                        cond=dirty_noisy_combined,
                        shape=shape,
                        device=device,
                        clip_denoised=True,
                        progress=False,
                        cond_fn=None,
                        model_kwargs=model_kwargs,
                    )[:actual_batch_size]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='./configs/generator.yaml',
                        help='Path to config')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder')
    parser.add_argument('--batch_size', '--bs', '-b', type=int,
                        default=50,
                        help='Batch size to use when generating samples')
    parser.add_argument('--guidance_scale', '-s', type=float,
                        default=3.)
    parser.add_argument('--timestep_respacing', '-t',
                        type=int, default=250)
    parser.add_argument('--runs_per_sample',
                        type=int, default=20)
    args = parser.parse_args()
    main(args)
