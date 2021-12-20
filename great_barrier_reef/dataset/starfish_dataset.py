from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches
from ast import literal_eval
import albumentations as A
import torch


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


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_coco_bboxes(
    plot_ax, bboxes, get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


class StarfishDataset(Dataset):
    def __init__(
        self,
        annotations_dataframe: pd.DataFrame,
        transforms: A.Compose,
        images_dir_path: str = "../data/train_images/",
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.image_paths = self.prepare_image_paths()
        self.annotations = self.prepare_annotations()
        self.transforms = transforms

    def prepare_image_paths(self) -> np.ndarray:
        image_paths = self.annotations_df.apply(
            lambda x: "video_{}/{}.jpg".format(x["video_id"], x["video_frame"]), axis=1
        ).values
        return image_paths

    def prepare_annotations(self) -> np.ndarray:
        annotations = self.annotations_df["annotations"].apply(literal_eval).values
        return annotations

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.image_paths[index]
        image = PIL.Image.open(self.images_dir_path / image_name)
        width, height = image.size
        bboxes, was_empty = get_bboxes_from_annotation(
            self.annotations[index], width, height
        )
        class_labels = np.zeros(1) if was_empty else np.ones(len(bboxes))

        return image, bboxes, class_labels, image_name

    def show_image(self, index):
        """get image by its index and plot."""
        image, bboxes, class_labels, image_name = self.get_image_and_labels_by_idx(
            index
        )
        print(f"image_name: {image_name}, index: {index}")
        self._show_image(image, bboxes)

    def _show_image(self, image, bboxes=None, figsize=(10, 10)):
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)
        if bboxes is not None:
            draw_coco_bboxes(ax, bboxes)

        plt.show()

    def __getitem__(self, index):
        image, bboxes, class_labels, image_name = self.get_image_and_labels_by_idx(
            index
        )
        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": bboxes,
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
        ]  # convert to yxyx - I have no idea

        target = {
            "bboxes": torch.as_tensor(pascal_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_name": image_name,
            "image_index": index,
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_name
