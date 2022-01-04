from great_barrier_reef import (
    StarfishDataset,
    StarfishDatasetAdapter,
    StarfishDataModule,
    StarfishEfficientDetModel,
    compare_bboxes_for_image,
)
from great_barrier_reef.utils import transforms
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger, NeptuneLogger
from pytorch_lightning.utilities.seed import seed_everything
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from omegaconf import OmegaConf


def train(config):
    seed_everything(config.seed)
    train_df = pd.read_csv(
        f"{config.data.data_folder}/train_fold{config.data.fold}.csv"
    )
    val_df = pd.read_csv(f"{config.data.data_folder}/val_fold{config.data.fold}.csv")

    adapter_dataset_train = StarfishDatasetAdapter(
        train_df,
        images_dir_path=f"{config.data.data_folder}/train_images/",
        keep_empty=config.data.keep_empty_images,
        apply_empty_aug=config.data.apply_empty_aug,
        mosaic_augmentation=config.data.mosaic_augmentation,
        **config.data.augmentator_args,
    )
    adapter_dataset_val = StarfishDatasetAdapter(
        val_df,
        images_dir_path=f"{config.data.data_folder}/train_images/",
        keep_empty=False,
        apply_empty_aug=False,
    )

    datamodule = StarfishDataModule(
        adapter_dataset_train,
        adapter_dataset_val,
        train_transforms=getattr(transforms, config.data.transforms)(
            target_img_size=config.data.target_img_size
        ),
        valid_transforms=getattr(transforms, "valid_transforms")(
            target_img_size=config.data.target_img_size
        ),
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
    )

    model = StarfishEfficientDetModel(
        num_classes=1,
        img_size=config.data.target_img_size,
        inference_transforms=getattr(transforms, "valid_transforms")(
            target_img_size=config.data.target_img_size
        ),
        model_architecture=f"{config.model_name}",
        learning_rate=config.lr,
        config=config,
    )

    callbacks = [
        EarlyStopping(
            monitor="valid_loss_epoch", patience=config.general.early_stopping
        ),
        ModelCheckpoint(
            verbose=True,
            monitor="valid_loss_epoch",
            dirpath=f"{config.general.checkpoint_path}/{config.experiment_name}",
            filename="{epoch}-{valid_loss_epoch:.2f}",
        ),
        LearningRateMonitor(),
    ]
    loggers = [
        CSVLogger(
            save_dir=f"{config.general.cvs_logs_path}", name=f"{config.experiment_name}"
        ),
        NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project_name="azkalot1/reef",
            experiment_name=f"{config.experiment_name}",
        ),
    ]
    trainer = Trainer(callbacks=callbacks, logger=loggers, **config.trainer_args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    conf = OmegaConf.load("default_config.yaml")
    conf_cli = OmegaConf.from_cli()
    result_config = OmegaConf.merge(conf, conf_cli)
    train(result_config)
