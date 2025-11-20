import argparse
from datetime import datetime
from pathlib import Path
import yaml

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from src.trainer import GeneratorModule
from src.utils import get_config


def main(args) -> None:
    config_path = args.config
    config = get_config(config_path)
    auto_lr = args.auto_lr
    auto_bs = args.auto_bs

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=f"{date_time}_{config['comment']}")

    precision = 16 if config['fp16'] else 32
    accumulate_grad_batches = 1 if not config['accumulate_grad_batches'] else config['accumulate_grad_batches']
    iterations = config['iterations']
    eval_every = config['eval_every']

    module = GeneratorModule(config, config['fp16'])
    
    # 构建 callbacks 列表
    callbacks = [
        LearningRateMonitor('step'),
        ModelCheckpoint(every_n_val_epochs=1, filename='last_{epoch}_{step}'),
        ModelCheckpoint(
            every_n_val_epochs=1, 
            filename='best_{epoch}_{step}', 
            monitor='val/weighted_loss',
            mode='min'
        )
    ]
    
    # 根据配置文件决定是否添加 early stopping
    early_stop_config = config.get('early_stopping', {})
    if early_stop_config.get('enabled', False):
        callback_early_stop = EarlyStopping(
            monitor=early_stop_config.get('monitor', 'val/weighted_loss'),
            patience=early_stop_config.get('patience', 10),
            mode='min',
            verbose=True,
            min_delta=early_stop_config.get('min_delta', 0.001),
        )
        callbacks.append(callback_early_stop)
        print(f"✓ Early Stopping enabled - patience={early_stop_config.get('patience', 10)}, "
              f"monitor={early_stop_config.get('monitor', 'val/weighted_loss')}")
    
    # 获取checkpoint路径
    path_checkpoint = config.get('fine_tune_from', None)
    if path_checkpoint:
        print(f"Resume from checkpoint: {path_checkpoint}")

    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         resume_from_checkpoint=path_checkpoint,
                         gpus=-1,
                         auto_select_gpus=True,
                         auto_scale_batch_size=auto_bs,
                         max_steps=iterations,
                         check_val_every_n_epoch=eval_every,
                         # strategy='ddp',
                         precision=precision,
                         accumulate_grad_batches=accumulate_grad_batches)

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(module, min_lr=1e-5, max_lr=1e-1,)
        lr = lr_finder.suggestion['lr']
        print(f"Suggested learning rate: {lr}")
        module.hparams.lr = lr
        config['lr_suggested'] = lr

    if auto_bs:
        trainer.tune(module)

    # 开始训练
    trainer.fit(module)

    # save config to file
    save_path = Path(logger.experiment.get_logdir()) / Path(config_path).name
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/generator.yaml', help='path to config file')
    parser.add_argument('--auto_bs', action='store_true', help='auto select batch size')
    parser.add_argument('--auto_lr', action='store_true', help='auto select learning rate')
    args = parser.parse_args()
    main(args)