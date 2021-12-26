from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
from ast import literal_eval
import albumentations as A
import torch
from great_barrier_reef.utils import get_valid_transforms, draw_pascal_voc_bboxes


def get_bboxes_from_annotation(annotation: list, im_width: int, im_height: int):
    if len(annotation) == 0:
        return [[0, 0, 1, 1]], True
    bboxes = []
    # since we have our annotations in COCO (x_min, y_min, width, height),
    # we need to convert in in pascal_voc
    for ann in annotation:
        bboxes.append(
            [
                ann["x"],
                ann["y"],
                min(ann["x"] + ann["width"], im_width),
                min(ann["y"] + ann["height"], im_height),
            ]
        )
    return bboxes, False


def get_area(annotation):
    total_bbox_area_images = 0
    for ann in annotation:
        total_bbox_area_images += ann["width"] * ann["height"]
    return total_bbox_area_images


class StarfishDatasetAdapter(object):
    def __init__(self, annotations_dataframe, images_dir_path="../data/train_images/"):
        self.annotations_df = annotations_dataframe
        self.images = self.prepare_image_ids()
        self.images_dir_path = Path(images_dir_path)
        self.image_paths = self.prepare_image_paths()
        self.annotations = self.prepare_annotations()
        self.total_areas = self.prepare_total_areas()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_path = self.image_paths[index]
        # image_names = self.images[index]
        image = PIL.Image.open(self.images_dir_path / image_path)
        width, height = image.size
        bboxes, image_is_empty = get_bboxes_from_annotation(
            self.annotations[index], width, height
        )
        class_labels = np.ones(len(bboxes))

        return image, bboxes, class_labels, index, image_is_empty

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
        image, bboxes, class_labels, image_id, image_is_empty = self.get_image_and_labels_by_idx(index)
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
    def __init__(self, dataset_adaptor, transforms=get_valid_transforms()):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
            image_is_empty
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
            :, [1, 0, 3, 2]
        ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
            "image_is_empty": torch.tensor([1]).int() if image_is_empty else torch.tensor([0])
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)
