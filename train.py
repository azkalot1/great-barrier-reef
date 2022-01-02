from great_barrier_reef import (
    StarfishDataset,
    StarfishDatasetAdapter,
    StarfishDataModule,
    StarfishEfficientDetModel,
    get_train_transforms_simple,
    get_train_transforms_super_heavy,
    get_train_transforms_heavy,
    get_valid_transforms,
    compare_bboxes_for_image,
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
    train_df = data_df.loc[data_df["video_id"] != 2, :]
    val_df = data_df.loc[data_df["video_id"] == 2, :]

    adapter_dataset_train = StarfishDatasetAdapter(
        train_df,
        images_dir_path="data/train_images/",
        keep_empty=True,
        apply_empty_aug=True,
        saved_crops_path="data/crops_val_fold2.pickle",
        min_insert_starfish=7,
    )
    adapter_dataset_val = StarfishDatasetAdapter(
        val_df,
        images_dir_path="data/train_images/",
        keep_empty=False,
        apply_empty_aug=False,
    )

    datamodule = StarfishDataModule(
        adapter_dataset_train,
        adapter_dataset_val,
        train_transforms=get_train_transforms_heavy(target_img_size=640),
        valid_transforms=get_valid_transforms(target_img_size=640),
        num_workers=8,
        batch_size=16,
    )

    model = StarfishEfficientDetModel(
        num_classes=1,
        img_size=640,
        inference_transforms=get_valid_transforms(target_img_size=640),
        model_architecture="cspresdet50",
        learning_rate=1e-3,
    )

    callbacks = [
        EarlyStopping(monitor="valid_loss_epoch", patience=20),
        ModelCheckpoint(verbose=True, monitor="valid_loss_epoch"),
        LearningRateMonitor(),
    ]
    loggers = [
        CSVLogger(
            save_dir="csv_logs", name="cspresdet50_cosine_AugsHeavy_InsertEmpty_Size640"
        ),
        NeptuneLogger(
            api_key=NEPTUNE_API_TOKEN,
            project_name="azkalot1/reef",
            experiment_name="cspresdet50_cosine_AugsHeavy_InsertEmpty_Size640",
        ),
    ]
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        gpus=4,
        max_epochs=150,
        num_sanity_val_steps=1,
        precision=16,
        accumulate_grad_batches=64,
        benchmark=True,
        deterministic=True,
        accelerator="ddp",
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
