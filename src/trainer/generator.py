from typing import Dict
import copy
from itertools import chain

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.model import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    PriorNet,
    EncoderNet,
    LatentProjector,
    kl_divergence_diag_gaussians,
)
from src.model import create_named_schedule_sampler
from src.model.resample import LossAwareSampler
from src.data import MakeDataLoader
from src.utils import update_ema


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = True


class GeneratorModule(pl.LightningModule):

    def __init__(self, config: Dict, use_fp16: bool = False,
                 timestep_respacing=None,):
        super().__init__()
        self.config = config

        # load data
        path_image = config['dataset']['image_path']
        path_labels = config['dataset']['label_path']
        real_data = config['dataset']['real_data']
        self.size_image = config['dataset']['size']
        self.n_channels = config['dataset']['n_channels']
        self.clip_denoised = config['clip_denoised']

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
            raise ValueError("latent.project_channels 必须为正数，否则无法拼接潜变量特征。")
        self.latent_prior_cfg = latent_cfg.get('prior', {})
        self.latent_encoder_cfg = latent_cfg.get('encoder', {})

        use_zeros = config['dataset']['use_zeros']
        from_uv = config['dataset']['from_uv']
        power = config['dataset']['power']


        self.make_dl = MakeDataLoader(folder_images=path_image,
                                      image_size=self.size_image, real_data=real_data,
                                      power=power, use_zeros=use_zeros, from_uv=from_uv
                                      )

        # training parameters
        self.hparams.batch_size = config['batch_size']  # set start batch_size
        self.hparams.lr = eval(config['lr'])

        # load model and diffusion
        params = model_and_diffusion_defaults()
        params['image_size'] = self.size_image
        params['n_classes'] = config['dataset']['n_classes']
        params['use_fp16'] = use_fp16
        base_in_channels = 2 * self.n_channels
        params['in_channels'] = base_in_channels + self.latent_channels if self.latent_enabled else base_in_channels
        params['out_channels'] = config['dataset']['n_channels']
        params['use_y_conditioning'] = config['model']['use_y_conditioning']
        params['timestep_respacing'] = timestep_respacing
        params['diffusion_steps'] = config['model']['diffusion_steps']

        self.model, self.diffusion = create_model_and_diffusion(**params)

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
        else:
            self.prior_net = None
            self.encoder_net = None
            self.latent_projector = None

        from src.model import init_weights
        init_weights(self.model)
        if self.latent_enabled:
            init_weights(self.prior_net)
            init_weights(self.encoder_net)
            init_weights(self.latent_projector)
        print("Applied custom weight initialization to main model")

        self._prior_frozen = False
        if self.latent_enabled and self.prior_freeze_steps > 0:
            self._set_prior_requires_grad(False)
            self._prior_frozen = True

        self.ema_rate = config['ema_rate']
        self.model_ema = copy.deepcopy(self.model).eval()

        if use_fp16:
            self.model.convert_to_fp16()

        # load sampler
        self.schedule_sampler = create_named_schedule_sampler(
            config['schedule_sampler'], self.diffusion
        )

    def get_dataloader(self, mode: str) -> DataLoader:
        """Returns dataloader

        :param mode: type of dataloader to return. Choices: train, val, test
        :return: dataloader
        """

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

    def _set_prior_requires_grad(self, requires_grad: bool) -> None:
        if self.prior_net is None:
            return
        for param in self.prior_net.parameters():
            param.requires_grad = requires_grad

    def _kl_beta(self) -> float:
        if (not self.latent_enabled) or self.kl_beta_max <= 0.0:
            return 0.0
        if self.kl_warmup_steps <= 0:
            return self.kl_beta_max
        progress = min(1.0, (self.global_step + 1) / float(self.kl_warmup_steps))
        return self.kl_beta_max * progress

    def _latent_forward(self, x0: torch.Tensor, z_dirty: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def configure_optimizers(self):
        lr = self.hparams.lr
        wd = eval(self.config['wd'])
        params = [self.model.parameters()]
        if self.latent_enabled:
            params.extend(
                [
                    self.prior_net.parameters(),
                    self.encoder_net.parameters(),
                    self.latent_projector.parameters(),
                ]
            )
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
        """Override to use set_to_none=True for better performance"""
        optimizer.zero_grad(set_to_none=True)

    def step(self, batch, batch_idx: int, *, stage: str) -> torch.Tensor:
        data = batch
        im = data["true"]
        dirty_noisy = data["dirty_noisy"]

        # 检查输入数据是否有NaN或Inf
        if torch.isnan(im).any() or torch.isinf(im).any():
            print(f"Warning: {stage} input 'im' contains NaN or Inf values")
            im = torch.nan_to_num(im, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.isnan(dirty_noisy).any() or torch.isinf(dirty_noisy).any():
            print(f"Warning: {stage} input 'dirty_noisy' contains NaN or Inf values")
            dirty_noisy = torch.nan_to_num(dirty_noisy, nan=0.0, posinf=1e6, neginf=-1e6)

        input = im

        model_kwargs = {}
        kl_loss = im.new_tensor(0.0)
        kl_per_sample = None
        kl_per_dim = None
        if self.latent_enabled:
            latent_outputs = self._latent_forward(im, dirty_noisy)
            model_kwargs["latent"] = latent_outputs["latent_map"]
            kl_per_sample, kl_per_dim = kl_divergence_diag_gaussians(
                mu_q=latent_outputs["mu_q"],
                log_var_q=latent_outputs["log_var_q"],
                mu_p=latent_outputs["mu_p"],
                log_var_p=latent_outputs["log_var_p"],
                free_bits=self.kl_free_bits,
            )
            kl_loss = kl_per_sample.mean()

        t, weights = self.schedule_sampler.sample(input.shape[0], device=self.device)
        losses = self.diffusion.training_losses(
            self.model,
            input,
            t,
            cond=dirty_noisy,
            model_kwargs=model_kwargs,
        )

        # 检查loss是否有NaN或Inf
        for k, v in losses.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                print(f"Warning: {stage} loss '{k}' contains NaN or Inf values")
                losses[k] = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses['loss'].detach()
            )

        diff_loss = (losses['loss'] * weights).mean()
        beta = self._kl_beta()
        total_loss = diff_loss + diff_loss.new_tensor(beta) * kl_loss
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Error: {stage} final loss is NaN or Inf, replacing with 0.0")
            total_loss = torch.tensor(0.0, device=diff_loss.device, requires_grad=True)
        
        res_dict = {
            f'{stage}/weighted_loss': total_loss.detach(),
            f'{stage}/diff_loss': diff_loss.detach(),
        }
        for k, v in losses.items():
            res_dict[f'{stage}/{k}'] = v.mean()
        if self.latent_enabled:
            res_dict[f'{stage}/kl_loss'] = kl_loss.detach()
            res_dict[f'{stage}/kl_beta'] = beta
            if kl_per_sample is not None:
                res_dict[f'{stage}/kl_per_sample'] = kl_per_sample.mean().detach()
            if kl_per_dim is not None:
                res_dict[f'{stage}/kl_per_dim'] = kl_per_dim.mean().detach()
        self.log_dict(res_dict)
        return total_loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage='val')

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """Unfreeze prior network after specified steps"""
        if (
            self.latent_enabled
            and self.prior_freeze_steps > 0
            and self._prior_frozen
            and self.global_step >= self.prior_freeze_steps
        ):
            self._set_prior_requires_grad(True)
            self._prior_frozen = False
            # Only print on main process in multi-GPU training
            if self.trainer.is_global_zero:
                print(f"Unfreezing prior network at step {self.global_step}")

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """Update EMA after each training batch"""
        self._update_ema()

    def on_after_backward(self) -> None:
        """Called after backward pass - kept for compatibility"""
        pass

    def _update_ema(self) -> None:
        """Update EMA model parameters"""
        # Only update EMA during training, not validation
        if self.training:
            update_ema(self.model_ema.parameters(), self.model.parameters(), rate=self.ema_rate)

    def forward(
        self,
        dirty_noisy: torch.Tensor,
        *,
        num_latent_samples: int = 1,
        denoised_fn=None,
        cond_fn=None,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        推理接口：给定脏图 dirty_noisy，采样潜变量并运行扩散逆过程。
        :param dirty_noisy: 条件输入，形状为 [B, C, H, W]。
        :param num_latent_samples: 采样的潜变量数量，>1 时返回 [B, M, C, H, W]。
        """
        if dirty_noisy.dim() != 4:
            raise ValueError("dirty_noisy 需要为四维张量 [B, C, H, W]")

        b = dirty_noisy.shape[0]
        shape = (b, self.n_channels, self.size_image, self.size_image)
        spatial_size = dirty_noisy.shape[-2:]

        samples = []
        for _ in range(num_latent_samples):
            model_kwargs = {}
            if self.latent_enabled:
                mu_p, log_var_p = self.prior_net(dirty_noisy)
                std_p = torch.exp(0.5 * log_var_p)
                z = mu_p + std_p * torch.randn_like(std_p)
                latent_map = self.latent_projector(z, spatial_size=spatial_size)
                model_kwargs["latent"] = latent_map

            generated = self.diffusion.p_sample_loop(
                self.model_ema,
                shape,
                cond=dirty_noisy,
                clip_denoised=self.clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                progress=progress,
            )
            samples.append(generated)

        if num_latent_samples == 1:
            return samples[0]
        return torch.stack(samples, dim=1)
