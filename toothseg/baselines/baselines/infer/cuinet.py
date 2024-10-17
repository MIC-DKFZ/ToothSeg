import argparse
from pathlib import Path

import os, sys
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from baselines.datamodules import CuiNetFovCropSegDataModule
from baselines.models import CuiNet


def predict(stage: str, devices: int):
    with open('baselines/config/cuinet.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    dm = CuiNetFovCropSegDataModule(
        seed=config['seed'], **config['datamodule'],
    )
    
    return_type = config['model'].pop('return_type')
    model = CuiNet(
        **config['model'],
        return_type=return_type if stage == 'multi_class' else stage,
        out_dir=list(map(Path, {
            'binary': ['roiPr'],
            'instances': ['masksPr', 'centroidsPr', 'skeletonsPr'],
            'multi_class': ['cuinetPr'],
        }[stage])),
    )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'cuinet_{config["version"]}',
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
    parser.add_argument('stage', choices=['binary', 'instances', 'multi_class'])
    parser.add_argument('--devices', required=False, default=1, type=int)
    args = parser.parse_args()

    predict(args.stage, args.devices)
