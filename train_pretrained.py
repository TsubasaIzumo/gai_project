"""
Training script for HuggingFace pretrained diffusion models
"""
import argparse
import yaml
import os
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.trainer import PretrainedGeneratorModule


def main():
    parser = argparse.ArgumentParser(description='Train pretrained diffusion model')
    parser.add_argument('--config', type=str, default='configs/generator_pretrained.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--test_mode', action='store_true',
                        help='Test mode: train only a few steps to verify code')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config file: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test mode configuration
    if args.test_mode:
        print("\n⚠️ Test mode: training only 10 steps for verification\n")
        config['iterations'] = 10
        config['batch_size'] = 2
        config['n_workers'] = 0
    
    # Create model
    print("Creating pretrained model...")
    model = PretrainedGeneratorModule(config)
    
    # 配置callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath='checkpoints_pretrained',
        filename='pretrained-{epoch:02d}-{step}-{val/total_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.get('early_stopping', {}).get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=config['early_stopping'].get('monitor', 'val/total_loss'),
            patience=config['early_stopping'].get('patience', 5),
            min_delta=config['early_stopping'].get('min_delta', 0.001),
            mode='min',
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        print(f"✓ Early stopping enabled (patience={config['early_stopping']['patience']})")
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Logger
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    comment = config.get('comment', 'pretrained_run')
    logger = TensorBoardLogger(
        'lightning_logs',
        name=f'{timestamp}_{comment}',
    )
    
    # Create trainer
    print("\nConfiguring Trainer...")
    trainer = pl.Trainer(
        max_steps=config['iterations'],
        callbacks=callbacks,
        logger=logger,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy='ddp' if args.gpus > 1 else 'auto',
        precision=16 if config.get('fp16', False) else 32,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        log_every_n_steps=50,
        val_check_interval=config.get('eval_every', 1.0),
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Print configuration summary
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Pretrained model: {config['pretrained']['model_id']}")
    print(f"Freeze strategy: {config['pretrained']['freeze_strategy']}")
    if config['pretrained']['freeze_strategy'] == 'partial':
        print(f"Trainable modules: {config['pretrained']['trainable_modules']}")
    print(f"Variational inference: {'Enabled' if config['latent']['enabled'] else 'Disabled'}")
    if config['latent']['enabled']:
        print(f"  - Latent dimension: {config['latent']['dim']}")
        print(f"  - Project channels: {config['latent']['project_channels']}")
        print(f"  - KL beta: {config['latent']['beta_max']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max iterations: {config['iterations']}")
    print(f"Image size: {config['dataset']['size']}×{config['dataset']['size']}")
    print(f"Number of GPUs: {args.gpus}")
    print(f"Log directory: {logger.log_dir}")
    print("="*60 + "\n")
    
    # Start training
    print("Starting training...\n")
    try:
        trainer.fit(model, ckpt_path=args.resume)
        print("\n✓ Training completed!")
        print(f"Best model: {checkpoint_callback.best_model_path}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise


if __name__ == '__main__':
    main()
