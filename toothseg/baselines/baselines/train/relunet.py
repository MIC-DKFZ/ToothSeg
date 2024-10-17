import argparse
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import os, sys
sys.path.append(os.getcwd())

from baselines.datamodules import (
    ReluNetDownsampledSegDataModule,
    ReluNetTeethCropSegDataModule,
    ReluNetToothPatchSegDataModule,
)
from baselines.models import (
    DownsampledToothSegmentationNet,
    MulticlassToothSegmentationNet,
    SingleToothSegmentationNet,
)


def train(stage: str, devices: int, checkpoint: Optional[str]):
    with open('baselines/config/relunet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule'].update(config['datamodule'][stage])
    if stage == 'roi':
        dm = ReluNetDownsampledSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'multiclass':
        dm = ReluNetTeethCropSegDataModule(seed=config['seed'], **config['datamodule'])
    elif stage == 'single_tooth':
        dm = ReluNetToothPatchSegDataModule(seed=config['seed'], **config['datamodule'])
    
    config['model'][stage].pop('pretrained')
    if stage == 'roi':
        model = DownsampledToothSegmentationNet(**config['model'][stage])
    elif stage == 'multiclass':
        model = MulticlassToothSegmentationNet(**config['model'][stage])
    elif stage == 'single_tooth':
        model = SingleToothSegmentationNet(**config['model'][stage])

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'relunet_{stage}_{config["version"]}',
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='epoch',
        mode='max',
        filename='weights-{epoch:02d}',
    )
    loss_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='loss/val',
        filename='weights-{epoch:02d}',
    )
    metric_checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor='dice/val',
        mode='max',
        filename='weights-{epoch:02d}',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        max_epochs=config['model'][stage]['epochs'],
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            epoch_checkpoint_callback,
            loss_checkpoint_callback,
            metric_checkpoint_callback,
            LearningRateMonitor(),
        ],
        sync_batchnorm=True,
        accumulate_grad_batches=3 // devices,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint)


if __name__ == '__main__':
    print('TODO: implement data augmentations')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['roi', 'multiclass', 'single_tooth'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    args = parser.parse_args()

    train(args.stage, args.devices, args.checkpoint)
