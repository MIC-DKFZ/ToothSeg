import argparse
from pathlib import Path

import os, sys
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from baselines.datamodules import WangNetFovCropSegDataModule
from baselines.models import WangNet


def predict(stage: str, devices: int):
    with open('baselines/config/wangnet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    dm = WangNetFovCropSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )
    
    model = WangNet(
        **config['model'],
        out_dir=Path('dentalnetPr' if stage == 'instances' else 'wangnetPr'),
    )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'wangnet_{config["version"]}',
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.predict(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['instances', 'wangnet'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    args = parser.parse_args()

    predict(args.stage, args.devices)
