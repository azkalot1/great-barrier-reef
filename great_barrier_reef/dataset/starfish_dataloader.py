from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
from .starfish_dataset import StarfishDataset
from great_barrier_reef.utils import valid_transforms
import pandas as pd
import torch


class StarfishDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_adaptor,
        validation_dataset_adaptor,
        train_transforms=valid_transforms(target_img_size=512),
        valid_transforms=valid_transforms(target_img_size=512),
        num_workers=4,
        batch_size=8,
    ):

        super().__init__()
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataset(self) -> StarfishDataset:
        return StarfishDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> StarfishDataset:
        return StarfishDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()
        img_is_empty = torch.tensor(
            [target["image_is_empty"] for target in targets]
        ).unsqueeze(1)

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
            "img_is_empty": img_is_empty,
        }

        return images, annotations, targets, image_ids
