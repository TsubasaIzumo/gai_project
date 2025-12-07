"""
Pretrained Diffusion Model Trainer
Uses HuggingFace pretrained DDPM models combined with existing variational inference architecture
"""
from typing import Dict, List, Optional
import copy
from itertools import chain

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# HuggingFace diffusers
from diffusers import UNet2DModel, DDPMScheduler

# Reuse existing variational inference modules
from src.model import (
    PriorNet,
    EncoderNet,
    LatentProjector,
    kl_divergence_diag_gaussians,
)
from src.data import MakeDataLoader
from src.utils import update_ema


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class PretrainedGeneratorModule(pl.LightningModule):
    """
    Generator module using HuggingFace pretrained diffusion models
    Preserves existing variational inference architecture (PriorNet, EncoderNet, LatentProjector)
    """

    def __init__(self, config: Dict, use_fp16: bool = False):
        super().__init__()
        self.config = config
        
        # Basic data configuration
        path_image = config['dataset']['image_path']
        path_labels = config['dataset']['label_path']
        real_data = config['dataset']['real_data']
        self.size_image = config['dataset']['size']
        self.n_channels = config['dataset']['n_channels']
        self.clip_denoised = config.get('clip_denoised', True)
        
        # Variational inference configuration
        latent_cfg = config.get('latent', {})
        self.latent_dim = int(latent_cfg.get('dim', 0) or 0)
        latent_enabled_flag = latent_cfg.get('enabled', False)
        self.latent_enabled = bool(latent_enabled_flag and self.latent_dim > 0)
        self.latent_channels = int(latent_cfg.get('project_channels', 0) or 0)
        self.kl_beta_max = float(latent_cfg.get('beta_max', 0.0) or 0.0)
        self.kl_warmup_steps = int(latent_cfg.get('beta_warmup_steps', 0) or 0)
        self.kl_free_bits = float(latent_cfg.get('free_bits', 0.0) or 0.0)
        self.prior_freeze_steps = int(latent_cfg.get('prior_freeze_steps', 0) or 0)

        if self.latent_enabled and self.latent_channels <= 0:
            raise ValueError("latent.project_channels must be positive to concatenate latent features.")
        
        self.latent_prior_cfg = latent_cfg.get('prior', {})
        self.latent_encoder_cfg = latent_cfg.get('encoder', {})
        
        # Pretrained model configuration
        pretrained_cfg = config.get('pretrained', {})
        self.pretrained_model_id = pretrained_cfg.get('model_id', 'google/ddpm-celebahq-256')
        self.freeze_strategy = pretrained_cfg.get('freeze_strategy', 'partial')  # 'all', 'partial', 'none'
        self.trainable_modules = pretrained_cfg.get('trainable_modules', ['conv_in', 'down_blocks.0'])
        
        # Lightweight model configuration (preset only)
        # Use pretrained_model_size to avoid conflict with PyTorch Lightning's internal attributes
        self.pretrained_model_size = pretrained_cfg.get('model_size', None)  # None, 'small', 'medium', 'large'
        
        # Data loader
        use_zeros = config['dataset']['use_zeros']
        from_uv = config['dataset']['from_uv']
        power = config['dataset']['power']
        
        self.make_dl = MakeDataLoader(
            folder_images=path_image,
            image_size=self.size_image, 
            real_data=real_data,
            power=power, 
            use_zeros=use_zeros, 
            from_uv=from_uv,
            max_samples=config['dataset'].get('max_samples', None)
        )
        
        # Training parameters
        self.hparams.batch_size = config['batch_size']
        self.hparams.lr = eval(config['lr'])
        
        # Calculate target input channels
        base_in_channels = 2 * self.n_channels  # x_t + dirty_noisy
        self.target_in_channels = base_in_channels + self.latent_channels if self.latent_enabled else base_in_channels
        
        # Load and adapt pretrained UNet model
        print(f"Loading pretrained model from {self.pretrained_model_id}...")
        self.model = self._load_and_adapt_pretrained_unet()
        print(f"Model loaded successfully. Input channels: {self.target_in_channels}")
        
        # Create scheduler (for training and sampling)
        diffusion_steps = config.get('model', {}).get('diffusion_steps', 1000)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="linear",
            prediction_type="epsilon",  # Predict noise
            clip_sample=False,  # We'll clip manually during sampling
        )
        
        # Apply freeze strategy
        self._apply_freeze_strategy()
        
        # Create variational inference modules (if enabled)
        if self.latent_enabled:
            prior_defaults = {
                "base_channels": 32,
                "num_layers": 3,
                "max_channels": 256,
            }
            prior_defaults.update(self.latent_prior_cfg)
            encoder_defaults = {
                "base_channels": 32,
                "num_layers": 3,
                "max_channels": 256,
            }
            encoder_defaults.update(self.latent_encoder_cfg)

            self.prior_net = PriorNet(
                in_channels=self.n_channels,
                latent_dim=self.latent_dim,
                base_channels=prior_defaults["base_channels"],
                num_layers=prior_defaults["num_layers"],
                max_channels=prior_defaults["max_channels"],
            )
            self.encoder_net = EncoderNet(
                in_channels_x=self.n_channels,
                in_channels_dirty=self.n_channels,
                latent_dim=self.latent_dim,
                base_channels=encoder_defaults["base_channels"],
                num_layers=encoder_defaults["num_layers"],
                max_channels=encoder_defaults["max_channels"],
            )
            self.latent_projector = LatentProjector(self.latent_dim, self.latent_channels)
            
            # Initialize variational inference module weights
            from src.model import init_weights
            init_weights(self.prior_net)
            init_weights(self.encoder_net)
            init_weights(self.latent_projector)
            print("Variational inference modules initialized")
        else:
            self.prior_net = None
            self.encoder_net = None
            self.latent_projector = None
        
        # Prior freeze state
        self._prior_frozen = False
        if self.latent_enabled and self.prior_freeze_steps > 0:
            self._set_prior_requires_grad(False)
            self._prior_frozen = True
        
        # EMA model
        self.ema_rate = config.get('ema_rate', 0.999)
        self.model_ema = copy.deepcopy(self.model).eval()
        
        if use_fp16:
            self.model = self.model.half()
            self.model_ema = self.model_ema.half()

    def _get_model_size_config(self, original_config) -> Optional[Dict]:
        """
        Get model configuration based on model_size preset
        
        Returns:
            Dict with block_out_channels, layers_per_block, attention_head_dim, or None if using original model
        """
        if self.pretrained_model_size is None:
            # Use original model
            return None
        
        # Use preset based on original config
        original_channels = list(original_config.block_out_channels)
        
        if self.pretrained_model_size == "small":
            # Reduce channels by ~75%: halve all channels, reduce layers
            block_out_channels = tuple([c // 2 for c in original_channels])
            layers_per_block = 1
            # Reduce attention head dim if available
            if hasattr(original_config, 'attention_head_dim') and original_config.attention_head_dim is not None:
                attention_head_dim = max(4, original_config.attention_head_dim // 2)
            else:
                attention_head_dim = None
        elif self.pretrained_model_size == "medium":
            # Reduce channels by ~50%: halve all channels, keep layers
            block_out_channels = tuple([c // 2 for c in original_channels])
            layers_per_block = original_config.layers_per_block
            attention_head_dim = original_config.attention_head_dim if hasattr(original_config, 'attention_head_dim') else None
        elif self.pretrained_model_size == "large":
            # Use original model - return None to indicate no changes needed
            return None
        else:
            raise ValueError(f"Unknown model_size: {self.pretrained_model_size}. Must be 'small', 'medium', or 'large'")
        
        return {
            'block_out_channels': block_out_channels,
            'layers_per_block': layers_per_block,
            'attention_head_dim': attention_head_dim
        }
 
    def _load_and_adapt_pretrained_unet(self) -> UNet2DModel:
        """
        Load pretrained UNet model and adapt input channels
        Supports lightweight model configuration via model_size preset
        """
        # 1. Load pretrained model
        pretrained_unet = UNet2DModel.from_pretrained(self.pretrained_model_id)
        pretrained_in_channels = pretrained_unet.config.in_channels
        
        print(f"Pretrained model input channels: {pretrained_in_channels}")
        print(f"Target input channels: {self.target_in_channels}")
        
        # 2. Check if we need lightweight model
        lightweight_config = self._get_model_size_config(pretrained_unet.config)
        use_lightweight = lightweight_config is not None
        
        if not use_lightweight and pretrained_in_channels == self.target_in_channels:
            # No lightweight config and channels match, return directly
            return pretrained_unet
        
        # 3. Create new model with adapted config
        # Use copy.deepcopy to avoid modifying the original config
        new_config = copy.deepcopy(pretrained_unet.config)
        new_config.in_channels = self.target_in_channels
        new_config.out_channels = self.n_channels  # Adapt output channels to match data
        new_config.sample_size = self.size_image  # Adapt image size
        
        # Apply lightweight configuration if specified
        if lightweight_config:
            print(f"Applying lightweight configuration: model_size={self.pretrained_model_size}")
            new_config.block_out_channels = lightweight_config['block_out_channels']
            new_config.layers_per_block = lightweight_config['layers_per_block']
            if lightweight_config.get('attention_head_dim') is not None:
                new_config.attention_head_dim = lightweight_config['attention_head_dim']
            print(f"  block_out_channels: {list(pretrained_unet.config.block_out_channels)} -> {list(new_config.block_out_channels)}")
            print(f"  layers_per_block: {pretrained_unet.config.layers_per_block} -> {new_config.layers_per_block}")
            if lightweight_config.get('attention_head_dim') is not None:
                print(f"  attention_head_dim: {pretrained_unet.config.attention_head_dim if hasattr(pretrained_unet.config, 'attention_head_dim') else 'N/A'} -> {new_config.attention_head_dim}")
        
        # Use from_config to create the model
        new_unet = UNet2DModel.from_config(new_config)
        
        # Print model size comparison
        pretrained_params = sum(p.numel() for p in pretrained_unet.parameters())
        new_params = sum(p.numel() for p in new_unet.parameters())
        reduction = (1 - new_params / pretrained_params) * 100
        print(f"Model size: {pretrained_params:,} -> {new_params:,} parameters ({reduction:.1f}% reduction)")
        
        # Verify and manually fix conv_in layer if needed
        # Sometimes from_config doesn't properly apply in_channels changes
        old_conv_in = pretrained_unet.conv_in
        new_conv_in = new_unet.conv_in
        
        print(f"  Debug: old_conv_in.in_channels={old_conv_in.in_channels}, new_conv_in.in_channels={new_conv_in.in_channels}")
        print(f"  Debug: old_conv_in.weight.shape={old_conv_in.weight.shape}, new_conv_in.weight.shape={new_conv_in.weight.shape}")
        
        # If conv_in still has wrong input channels, manually replace it
        conv_in_replaced = False
        if new_conv_in.in_channels != self.target_in_channels:
            print(f"  Warning: conv_in has {new_conv_in.in_channels} channels, expected {self.target_in_channels}. Manually replacing...")
            # Create new conv_in layer with correct input channels
            out_channels = new_conv_in.out_channels
            kernel_size = new_conv_in.kernel_size
            stride = new_conv_in.stride
            padding = new_conv_in.padding
            
            new_conv_in_layer = nn.Conv2d(
                in_channels=self.target_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            new_unet.conv_in = new_conv_in_layer
            conv_in_replaced = True
            print(f"  Created new conv_in layer: {new_conv_in_layer.weight.shape}")
        
        # Verify and manually fix conv_out layer if needed
        # Ensure output channels match our data channels
        old_conv_out = pretrained_unet.conv_out
        new_conv_out = new_unet.conv_out
        
        print(f"  Debug: old_conv_out.out_channels={old_conv_out.out_channels}, new_conv_out.out_channels={new_conv_out.out_channels}, target={self.n_channels}")
        
        # If conv_out still has wrong output channels, manually replace it
        if new_conv_out.out_channels != self.n_channels:
            print(f"  Warning: conv_out has {new_conv_out.out_channels} channels, expected {self.n_channels}. Manually replacing...")
            # Create new conv_out layer with correct output channels
            in_channels = new_conv_out.in_channels
            kernel_size = new_conv_out.kernel_size
            stride = new_conv_out.stride
            padding = new_conv_out.padding
            
            new_conv_out_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.n_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            new_unet.conv_out = new_conv_out_layer
            print(f"  Created new conv_out layer: {new_conv_out_layer.weight.shape}")
        
        # 4. Transfer weights intelligently
        print("Transferring pretrained weights (intelligent transfer)...")
        with torch.no_grad():
            # Get pretrained model state_dict
            pretrained_state = pretrained_unet.state_dict()
            # Get new model state_dict (after potential conv_in replacement)
            new_state = new_unet.state_dict()
            
            # Statistics
            total_params = len(new_state)
            transferred_params = 0
            adapted_params = 0
            skipped_params = 0
            
            # Copy weights layer by layer
            for name, param in pretrained_state.items():
                if name not in new_state:
                    skipped_params += 1
                    continue
                
                new_param = new_state[name]
                
                # Case 1: Exact shape match - direct copy
                if param.shape == new_param.shape:
                    new_state[name] = param
                    transferred_params += 1
                
                # Case 2: conv_in layer (different input channels)
                elif 'conv_in.weight' in name:
                    old_weight = param  # [out_ch, in_ch_old, k, k]
                    new_weight = new_param  # [out_ch, in_ch_new, k, k]
                    
                    # Copy overlapping channels
                    min_out = min(old_weight.shape[0], new_weight.shape[0])
                    min_in = min(old_weight.shape[1], new_weight.shape[1])
                    new_weight[:min_out, :min_in, :, :] = old_weight[:min_out, :min_in, :, :]
                    
                    # Initialize new channels with small random values
                    if new_weight.shape[1] > min_in:
                        mean_val = old_weight.mean()
                        std_val = old_weight.std()
                        new_weight[:, min_in:, :, :] = torch.randn_like(
                            new_weight[:, min_in:, :, :]
                        ) * std_val * 0.1 + mean_val
                    
                    if new_weight.shape[0] > min_out:
                        mean_val = old_weight.mean()
                        std_val = old_weight.std()
                        new_weight[min_out:, :, :, :] = torch.randn_like(
                            new_weight[min_out:, :, :, :]
                        ) * std_val * 0.1 + mean_val
                    
                    new_state[name] = new_weight
                    adapted_params += 1
                
                # Case 3: conv_out layer (different output channels)
                elif 'conv_out.weight' in name:
                    old_weight = param  # [out_ch_old, in_ch, k, k]
                    new_weight = new_param  # [out_ch_new, in_ch, k, k]
                    
                    # Copy overlapping channels
                    min_out = min(old_weight.shape[0], new_weight.shape[0])
                    min_in = min(old_weight.shape[1], new_weight.shape[1])
                    new_weight[:min_out, :min_in, :, :] = old_weight[:min_out, :min_in, :, :]
                    
                    # Initialize new channels
                    if new_weight.shape[0] > min_out:
                        mean_val = old_weight.mean()
                        std_val = old_weight.std()
                        new_weight[min_out:, :, :, :] = torch.randn_like(
                            new_weight[min_out:, :, :, :]
                        ) * std_val * 0.1 + mean_val
                    
                    if new_weight.shape[1] > min_in:
                        mean_val = old_weight.mean()
                        std_val = old_weight.std()
                        new_weight[:, min_in:, :, :] = torch.randn_like(
                            new_weight[:, min_in:, :, :]
                        ) * std_val * 0.1 + mean_val
                    
                    new_state[name] = new_weight
                    adapted_params += 1
                
                # Case 4: conv_out bias
                elif 'conv_out.bias' in name:
                    old_bias = param  # [out_ch_old]
                    new_bias = new_param  # [out_ch_new]
                    
                    min_channels = min(old_bias.shape[0], new_bias.shape[0])
                    new_bias[:min_channels] = old_bias[:min_channels]
                    
                    if new_bias.shape[0] > min_channels:
                        new_bias[min_channels:] = 0.0
                    
                    new_state[name] = new_bias
                    adapted_params += 1
                
                # Case 5: Convolutional layers with channel mismatch (in down/up blocks)
                elif len(param.shape) == 4 and len(new_param.shape) == 4:  # Conv2d weights
                    # Format: [out_ch, in_ch, k, k]
                    old_out, old_in = param.shape[0], param.shape[1]
                    new_out, new_in = new_param.shape[0], new_param.shape[1]
                    
                    # Only transfer if new model channels are subset of old model channels
                    if new_out <= old_out and new_in <= old_in:
                        new_param[:, :, :, :] = param[:new_out, :new_in, :, :]
                        new_state[name] = new_param
                        adapted_params += 1
                    else:
                        # Shape mismatch too large, keep random initialization
                        skipped_params += 1
                
                # Case 6: Linear/1D layers (time embedding, etc.)
                elif len(param.shape) == 2 and len(new_param.shape) == 2:  # Linear weights
                    old_out, old_in = param.shape
                    new_out, new_in = new_param.shape
                    
                    # Transfer overlapping part
                    if new_out <= old_out and new_in <= old_in:
                        new_param[:, :] = param[:new_out, :new_in]
                        new_state[name] = new_param
                        adapted_params += 1
                    elif new_out >= old_out and new_in >= old_in:
                        # New model is larger, copy what we can
                        new_param[:old_out, :old_in] = param
                        # Initialize rest with small random values
                        mean_val = param.mean()
                        std_val = param.std()
                        if new_out > old_out:
                            new_param[old_out:, :] = torch.randn_like(
                                new_param[old_out:, :]
                            ) * std_val * 0.1 + mean_val
                        if new_in > old_in:
                            new_param[:, old_in:] = torch.randn_like(
                                new_param[:, old_in:]
                            ) * std_val * 0.1 + mean_val
                        new_state[name] = new_param
                        adapted_params += 1
                    else:
                        skipped_params += 1
                
                # Case 7: Bias terms (1D)
                elif len(param.shape) == 1 and len(new_param.shape) == 1:
                    min_len = min(param.shape[0], new_param.shape[0])
                    new_param[:min_len] = param[:min_len]
                    if new_param.shape[0] > min_len:
                        new_param[min_len:] = 0.0
                    new_state[name] = new_param
                    adapted_params += 1
                
                # Case 8: Other parameters - skip (keep random initialization)
                else:
                    skipped_params += 1
            
            # Load adapted weights
            new_unet.load_state_dict(new_state, strict=False)
            
            # Print statistics
            print(f"Weight transfer statistics:")
            print(f"  Total parameters: {total_params}")
            print(f"  Directly transferred: {transferred_params} ({100*transferred_params/total_params:.1f}%)")
            print(f"  Adapted (partial transfer): {adapted_params} ({100*adapted_params/total_params:.1f}%)")
            print(f"  Skipped (random init): {skipped_params} ({100*skipped_params/total_params:.1f}%)")
        
        print("Weight transfer completed!")
        return new_unet

    def _apply_freeze_strategy(self):
        """
        Apply model freezing strategy
        """
        if self.freeze_strategy == 'all':
            # Freeze all diffusion model parameters
            print("Freeze strategy: all (fully frozen)")
            for param in self.model.parameters():
                param.requires_grad = False
        
        elif self.freeze_strategy == 'partial':
            # Freeze all first, then unfreeze specified modules
            print(f"Freeze strategy: partial, trainable modules: {self.trainable_modules}")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze specified modules
            for module_name in self.trainable_modules:
                module = self._get_module_by_name(self.model, module_name)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"  Unfrozen module: {module_name}")
                else:
                    print(f"  Warning: Module {module_name} not found")
        
        elif self.freeze_strategy == 'none':
            # No freezing, all trainable
            print("Freeze strategy: none (all trainable)")
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Print trainable parameter statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Diffusion model parameters: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.1f}%)")

    def _get_module_by_name(self, model: nn.Module, module_name: str) -> Optional[nn.Module]:
        """
        Get module by name
        """
        parts = module_name.split('.')
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _set_prior_requires_grad(self, requires_grad: bool) -> None:
        """Set gradient requirement for prior network"""
        if self.prior_net is None:
            return
        for param in self.prior_net.parameters():
            param.requires_grad = requires_grad

    def _kl_beta(self) -> float:
        """Calculate current KL weight (with warmup)"""
        if (not self.latent_enabled) or self.kl_beta_max <= 0.0:
            return 0.0
        if self.kl_warmup_steps <= 0:
            return self.kl_beta_max
        progress = min(1.0, (self.global_step + 1) / float(self.kl_warmup_steps))
        return self.kl_beta_max * progress

    def _latent_forward(self, x0: torch.Tensor, z_dirty: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Variational inference forward pass"""
        mu_q, log_var_q = self.encoder_net(x0, z_dirty)
        mu_p, log_var_p = self.prior_net(z_dirty)
        std_q = torch.exp(0.5 * log_var_q)
        eps_q = torch.randn_like(std_q)
        z = mu_q + std_q * eps_q
        spatial_size = x0.shape[-2:]
        latent_map = self.latent_projector(z, spatial_size=spatial_size)
        return {
            "z": z,
            "latent_map": latent_map,
            "mu_q": mu_q,
            "log_var_q": log_var_q,
            "mu_p": mu_p,
            "log_var_p": log_var_p,
        }

    def get_dataloader(self, mode: str) -> DataLoader:
        """Return dataloader"""
        bs = self.hparams.batch_size
        n_workers = self.config['n_workers']

        if mode == 'train':
            return self.make_dl.get_data_loader_train(batch_size=bs, shuffle=True, num_workers=n_workers)
        elif mode == 'val':
            return self.make_dl.get_data_loader_valid(batch_size=bs, shuffle=False, num_workers=n_workers)
        elif mode == 'test':
            return self.make_dl.get_data_loader_test(batch_size=bs, shuffle=False, num_workers=n_workers)
        else:
            raise ValueError('mode must be one of train, val, test')

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader('test')

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])
        
        # Collect all parameters to optimize
        params = []
        
        # Trainable parameters from diffusion model
        diffusion_params = [p for p in self.model.parameters() if p.requires_grad]
        if diffusion_params:
            params.append(diffusion_params)
        
        # Variational inference module parameters
        if self.latent_enabled:
            params.extend([
                self.prior_net.parameters(),
                self.encoder_net.parameters(),
                self.latent_projector.parameters(),
            ])
        
        opt = optim.AdamW(chain.from_iterable(params), lr=lr, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config['iterations'], eta_min=lr / 10
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx=0):
        """Use set_to_none=True for better performance"""
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        """Training/validation step"""
        data = batch
        im = data["true"]
        dirty_noisy = data["dirty_noisy"]

        # Check input data
        if torch.isnan(im).any() or torch.isinf(im).any():
            print(f"Warning: {stage} input 'im' contains NaN or Inf values")
            im = torch.nan_to_num(im, nan=0.0, posinf=10.0, neginf=-10.0)
            im = torch.clamp(im, -10, 10)
        if torch.isnan(dirty_noisy).any() or torch.isinf(dirty_noisy).any():
            print(f"Warning: {stage} input 'dirty_noisy' contains NaN or Inf values")
            dirty_noisy = torch.nan_to_num(dirty_noisy, nan=0.0, posinf=10.0, neginf=-10.0)
            dirty_noisy = torch.clamp(dirty_noisy, -10, 10)

        x_start = im  # Target clean image
        batch_size = x_start.shape[0]

        # Variational inference (if enabled)
        kl_loss = x_start.new_tensor(0.0)
        latent_map = None
        
        if self.latent_enabled:
            latent_outputs = self._latent_forward(x_start, dirty_noisy)
            latent_map = latent_outputs["latent_map"]
            kl_per_sample, kl_per_dim = kl_divergence_diag_gaussians(
                mu_q=latent_outputs["mu_q"],
                log_var_q=latent_outputs["log_var_q"],
                mu_p=latent_outputs["mu_p"],
                log_var_p=latent_outputs["log_var_p"],
                free_bits=self.kl_free_bits,
            )
            kl_loss = kl_per_sample.mean()

        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # Add noise
        noise = torch.randn_like(x_start)
        noisy_x = self.scheduler.add_noise(x_start, noise, timesteps)

        # Build model input
        model_inputs = [noisy_x, dirty_noisy]
        if latent_map is not None:
            model_inputs.append(latent_map)
        model_input = torch.cat(model_inputs, dim=1)

        # Model prediction
        model_output = self.model(model_input, timesteps).sample

        # Calculate diffusion loss
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "sample":
            target = x_start
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x_start, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction_type: {self.scheduler.config.prediction_type}")

        diff_loss = F.mse_loss(model_output, target)

        # Total loss
        beta = self._kl_beta()
        total_loss = diff_loss + beta * kl_loss

        # Check final loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Error: {stage} final loss is NaN or Inf, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=diff_loss.device, requires_grad=True)

        # Log metrics
        res_dict = {
            f'{stage}/total_loss': total_loss.detach(),
            f'{stage}/diff_loss': diff_loss.detach(),
        }
        
        if self.latent_enabled:
            res_dict[f'{stage}/kl_loss'] = kl_loss.detach()
            res_dict[f'{stage}/kl_beta'] = beta

        # Logging
        if stage == 'train':
            self.log_dict(res_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:
            self.log_dict(res_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """Check if prior network should be unfrozen at training batch start"""
        if (
            self.latent_enabled
            and self.prior_freeze_steps > 0
            and self._prior_frozen
            and self.global_step >= self.prior_freeze_steps
        ):
            self._set_prior_requires_grad(True)
            self._prior_frozen = False
            if self.trainer.is_global_zero:
                print(f"Unfreezing prior network at step {self.global_step}")

    def on_train_start(self):
        """Ensure all models are on the correct device"""
        # PyTorch Lightning automatically moves registered modules, but we ensure model_ema is synced
        if self.model_ema is not None:
            # Ensure model_ema is on the same device as model
            model_device = next(self.model.parameters()).device
            model_ema_device = next(self.model_ema.parameters()).device
            if model_device != model_ema_device:
                self.model_ema = self.model_ema.to(model_device)
                if self.trainer.is_global_zero:
                    print(f"Model EMA moved to device: {model_device}")

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Update EMA model"""
        self._update_ema()

    def _update_ema(self) -> None:
        """Update EMA model parameters"""
        if self.training:
            # Ensure both models are on the same device
            model_device = next(self.model.parameters()).device
            model_ema_device = next(self.model_ema.parameters()).device
            if model_device != model_ema_device:
                self.model_ema = self.model_ema.to(model_device)
            update_ema(self.model_ema.parameters(), self.model.parameters(), rate=self.ema_rate)

    @torch.no_grad()
    def forward(
        self,
        dirty_noisy: torch.Tensor,
        *,
        num_latent_samples: int = 1,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Inference interface: Given dirty_noisy, sample latent variables and run diffusion reverse process
        
        Args:
            dirty_noisy: Conditional input, shape [B, C, H, W]
            num_latent_samples: Number of latent samples, >1 returns [B, M, C, H, W]
            progress: Whether to show progress bar
        
        Returns:
            Generated images, shape [B, C, H, W] or [B, M, C, H, W]
        """
        if dirty_noisy.dim() != 4:
            raise ValueError("dirty_noisy must be a 4D tensor [B, C, H, W]")

        b = dirty_noisy.shape[0]
        spatial_size = dirty_noisy.shape[-2:]
        device = dirty_noisy.device

        samples = []
        for _ in range(num_latent_samples):
            # Sample latent variables (if enabled)
            latent_map = None
            if self.latent_enabled:
                mu_p, log_var_p = self.prior_net(dirty_noisy)
                std_p = torch.exp(0.5 * log_var_p)
                z = mu_p + std_p * torch.randn_like(std_p)
                latent_map = self.latent_projector(z, spatial_size=spatial_size)

            # Start from pure noise
            image = torch.randn(
                b, self.n_channels, self.size_image, self.size_image,
                device=device
            )

            # Set scheduler to inference mode
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

            # Iterative denoising
            timesteps = self.scheduler.timesteps
            if progress:
                from tqdm.auto import tqdm
                timesteps = tqdm(timesteps, desc="Sampling")

            for t in timesteps:
                # Build model input
                model_inputs = [image, dirty_noisy]
                if latent_map is not None:
                    model_inputs.append(latent_map)
                model_input = torch.cat(model_inputs, dim=1)

                # Model prediction
                t_tensor = torch.tensor([t] * b, device=device)
                model_output = self.model_ema(model_input, t_tensor).sample
                
                # Ensure model_output has correct number of channels
                if model_output.shape[1] != self.n_channels:
                    # If output channels don't match, take first n_channels
                    model_output = model_output[:, :self.n_channels, :, :]

                # Scheduler step
                image = self.scheduler.step(model_output, t, image).prev_sample
                
                # Ensure image has correct number of channels (safety check)
                if image.shape[1] != self.n_channels:
                    image = image[:, :self.n_channels, :, :]

            # Clip if needed
            if self.clip_denoised:
                image = torch.clamp(image, -1, 1)

            samples.append(image)

        if num_latent_samples == 1:
            return samples[0]
        return torch.stack(samples, dim=1)

