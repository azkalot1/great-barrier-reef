from numbers import Number
from typing import List
from functools import singledispatch

import numpy as np
import torch

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.decorators import auto_move_data

import timm
import torchmetrics


def get_metrics(prefix, num_classes=1):
    rocuac = torchmetrics.AUROC(num_classes=1).cuda()
    f1 = torchmetrics.F1(num_classes=1).cuda()
    precision = torchmetrics.Precision(num_classes=1).cuda()
    recall = torchmetrics.Recall(num_classes=1).cuda()

    metrics_collection = torchmetrics.MetricCollection(
        [rocuac, f1, precision, recall], prefix=prefix
    )
    return metrics_collection


class StarfishClassifyIsEmptyModel(LightningModule):
    def __init__(
        self,
        num_classes=1,
        learning_rate=3e-4,
        model_architecture="tf_efficientnet_b0_ap",
        anchor_scale=4,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_architecture, pretrained=True, num_classes=num_classes
        )
        self.lr = learning_rate
        self.metrics_validation = get_metrics(num_classes=num_classes, prefix="valid_")
        self.metrics_train = get_metrics(num_classes=num_classes, prefix="train_")
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, verbose=True, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "valid_loss_epoch",
        }

    def training_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch

        y = annotations["img_is_empty"]
        yhat = self.model(images)

        loss = self.criterion(yhat, y.float())
        metrics = self.metrics_train(yhat, y)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch

        y = annotations["img_is_empty"]
        yhat = self.model(images)

        loss = self.criterion(yhat, y.float())
        metrics = self.metrics_validation(yhat, y)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "valid_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        self.log_dict(
            {k + "_epoch": v for k, v in self.metrics_validation.compute().items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def train_epoch_end(self, outputs):
        self.log_dict(
            {k + "_epoch": v for k, v in self.metrics_train.compute().items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
