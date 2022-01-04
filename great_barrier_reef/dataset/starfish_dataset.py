from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
from ast import literal_eval
import albumentations as A
import torch
from great_barrier_reef.utils import valid_transforms, draw_pascal_voc_bboxes
from .utils import get_bboxes_from_annotation, get_area, fix_bboxes
from .augmentation import ImageInsertAug, MosaicAugmentator


class StarfishDatasetAdapter(Dataset):
    def __init__(
        self,
        annotations_dataframe,
        images_dir_path="../data/train_images/",
        keep_empty=True,
        apply_empty_aug=False,
        mosaic_augmentation=False,
        **kwargs,
    ):
        self.keep_empty = keep_empty
        self.apply_empty_aug = apply_empty_aug
        if not self.keep_empty:
            # remove empty annotations
            annotations_dataframe = annotations_dataframe.loc[
                annotations_dataframe["annotations"] != "[]", :
            ]
        self.annotations_df = annotations_dataframe
        self.empty_augmentator = None
        if self.apply_empty_aug:
            # prepare augmentation of empty images
            assert (
                self.keep_empty is not False
            ), "Need to keep empty images to augment them"
            self.empty_augmentator = ImageInsertAug(
                non_empty_df=self.annotations_df.loc[
                    self.annotations_df["annotations"] != "[]", :
                ],
                images_dir_path=images_dir_path,
                **kwargs,
            )
        self.images = self.prepare_image_ids()
        self.images_dir_path = Path(images_dir_path)
        self.image_paths = self.prepare_image_paths()
        self.annotations = self.prepare_annotations()
        self.total_areas = self.prepare_total_areas()
        self.mosaic_augmentation = mosaic_augmentation
        self.get_element = (
            self.get_image_and_labels_by_idx_mosaic
            if self.mosaic_augmentation
            else self.get_image_and_labels_by_idx
        )
        if self.mosaic_augmentation:
            self.mosaic_augmentator = MosaicAugmentator()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(self.images_dir_path / image_path)
        width, height = image.size
        bboxes, image_is_empty = get_bboxes_from_annotation(
            self.annotations[index], width, height
        )
        if image_is_empty and self.apply_empty_aug:
            image, bboxes = self.empty_augmentator.augment_image(np.array(image))
            image = PIL.Image.fromarray(image)
            bboxes = fix_bboxes(bboxes, width, height)

        class_labels = np.ones(len(bboxes))

        return image, bboxes, class_labels, index, image_is_empty

    def get_image_and_labels_by_idx_mosaic(self, index):
        selected_idx = [index] + list(np.random.choice(len(self), 3))
        images = []
        bboxes = []
        for i in selected_idx:
            _image, _bboxes, _class_labels, _, _ = self.get_image_and_labels_by_idx(i)
            images.append(np.array(_image))
            bboxes.append(np.array(_bboxes))

        (
            image,
            bboxes,
            class_labels,
            image_is_empty,
        ) = self.mosaic_augmentator.run_augmentation(images, bboxes)

        return image, bboxes, class_labels, index, image_is_empty

    def __getitem__(self, index):
        return self.get_element(index)

    def prepare_image_paths(self) -> np.ndarray:
        image_paths = self.annotations_df.apply(
            lambda x: "video_{}/{}.jpg".format(x["video_id"], x["video_frame"]), axis=1
        ).values
        return image_paths

    def prepare_image_ids(self) -> np.ndarray:
        image_ids = self.annotations_df["image_id"].values
        return image_ids

    def prepare_annotations(self) -> np.ndarray:
        annotations = self.annotations_df["annotations"].apply(literal_eval).values
        return annotations

    def prepare_total_areas(self) -> np.ndarray:
        total_areas = [get_area(x) for x in self.annotations]
        return total_areas

    def show_image(self, index):
        (
            image,
            bboxes,
            class_labels,
            image_id,
            image_is_empty,
        ) = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}, is empty? {image_is_empty}")
        self._show_image(image, bboxes)
        print(class_labels)

    def _show_image(
        self,
        image,
        bboxes=None,
        draw_bboxes_fn=draw_pascal_voc_bboxes,
        figsize=(10, 10),
    ):
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        if bboxes is not None:
            draw_bboxes_fn(ax, bboxes)

        plt.show()


class StarfishDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=valid_transforms()):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
            image_is_empty,
        ) = self.ds[index]

        sample = {
            "image": np.array(image, dtype=np.uint8),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"], dtype=np.float32)
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        if len(sample["bboxes"]) > 0:
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
                :, [1, 0, 3, 2]
            ]  # convert to yxyx
        else:
            sample["bboxes"] = np.array([[0, 0, 1, 1]])
            labels = np.zeros(1)

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
            "image_is_empty": torch.tensor([1]).int()
            if image_is_empty
            else torch.tensor([0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)
