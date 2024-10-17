import argparse
from pathlib import Path

import os, sys
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from baselines.datamodules import LiuNetFovCropSegDataModule
from baselines.models import LiuNet


def predict(stage: str, devices: int):
    with open('baselines/config/liunet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    dm = LiuNetFovCropSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )
    
    return_type = config['model'].pop('return_type')
    model = LiuNet(
        **config['model'],
        return_type='binary' if stage == 'binary' else return_type,
        out_dir=Path('binaryPr' if stage == 'binary' else 'liunetPr'),
    )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'liunet_{config["version"]}',
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
    parser.add_argument('stage', choices=['binary', 'instances'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    args = parser.parse_args()

    predict(args.stage, args.devices)
