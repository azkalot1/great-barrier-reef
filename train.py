from great_barrier_reef import (
    StarfishDataset,
    StarfishDatasetAdapter,
    StarfishDataModule,
    StarfishEfficientDetModel,
    get_train_transforms,
    get_valid_transforms,
    compare_bboxes_for_image,
    get_train_transforms_pad,
    get_valid_transforms_pad,
)
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
import torch


def train():
    NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Yjg5NjBiZC02ZWJjLTQ2MWYtOWEzZi0wNDdiM2ZjMjdjNjMifQ=="
    data_df = pd.read_csv("data/train.csv")
    non_empty_df = data_df.loc[data_df["annotations"] != "[]", :]
    train_df = non_empty_df.loc[non_empty_df["video_id"] != 2, :]
    val_df = non_empty_df.loc[non_empty_df["video_id"] == 2, :]

    adapter_dataset_train = StarfishDatasetAdapter(
        train_df, images_dir_path="data/train_images/"
    )
    adapter_dataset_val = StarfishDatasetAdapter(
        val_df, images_dir_path="data/train_images/"
    )

    datamodule = StarfishDataModule(
        adapter_dataset_train,
        adapter_dataset_val,
        train_transforms=get_train_transforms(target_img_size=1280),
        valid_transforms=get_valid_transforms(target_img_size=1280),
        num_workers=8,
        batch_size=4,
    )

    model = StarfishEfficientDetModel(
        num_classes=1,
        img_size=1280,
        inference_transforms=get_valid_transforms(target_img_size=1280),
        model_architecture="tf_efficientdet_d0_ap",
        learning_rate=1e-2,
    )

    callbacks = [
        EarlyStopping(monitor="valid_loss_epoch", patience=15),
        ModelCheckpoint(verbose=True, monitor="valid_loss_epoch"),
        LearningRateMonitor(),
    ]
    loggers = [
        CSVLogger(save_dir="csv_logs", name="ddp_full3"),
        NeptuneLogger(
            api_key=NEPTUNE_API_TOKEN,
            project_name="azkalot1/reef",
            experiment_name="ddp_full3",
        ),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        gpus=4,
        max_epochs=150,
        num_sanity_val_steps=1,
        precision=16,
        accumulate_grad_batches=32,
        benchmark=True,
        deterministic=True,
        accelerator="ddp",
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
