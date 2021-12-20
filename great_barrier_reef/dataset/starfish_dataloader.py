from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
from .starfish_dataset import StarfishDataset
import pandas as pd
import torch


class StarfishDataModule(LightningDataModule):
    def __init__(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        train_transforms: A.Compose,
        validation_transforms: A.Compose,
        images_dir_path: str = "../data/train_images/",
        num_workers=4,
        batch_size=8,
    ):

        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.train_transforms = train_transforms
        self.validation_transforms = validation_transforms
        self.images_dir_path = images_dir_path
        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataset(self) -> StarfishDataset:
        return StarfishDataset(
            annotations_dataframe=self.train_data,
            transforms=self.train_transforms,
            images_dir_path=self.images_dir_path,
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = DataLoader(
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
            annotations_dataframe=self.validation_data,
            transforms=self.validation_transforms,
            images_dir_path=self.images_dir_path,
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = DataLoader(
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
        images, targets, image_names = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets
