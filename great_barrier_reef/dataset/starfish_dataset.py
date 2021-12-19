from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches
from ast import literal_eval


def get_bboxes_from_annotation(annotation: list):
    if len(annotation) == 0:
        return None

    bboxes = []
    for ann in annotation:
        bboxes.append([ann["x"], ann["y"], ann["width"], ann["height"]])
    return bboxes


def split_coco_bbox_fn(bbox):
    return bbox[:2], bbox[2], bbox[3]


def draw_coco_bboxes(plot_ax, bboxes, get_rectangle_corners_fn=split_coco_bbox_fn):
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


class StarfishDatasetAdaptor(Dataset):
    def __init__(
        self,
        annotations_dataframe: pd.DataFrame,
        images_dir_path: str = "../data/train_images/",
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.image_paths = self.prepare_image_paths()
        self.annotations = self.prepare_annotations()

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
        bboxes = get_bboxes_from_annotation(self.annotations[index])

        return image, bboxes, image_name

    def show_image(self, index):
        """get image by its index and plot."""
        image, bboxes, image_name = self.get_image_and_labels_by_idx(index)
        print(f"image_name: {image_name}")
        self._show_image(image, bboxes)

    def _show_image(self, image, bboxes=None, figsize=(10, 10)):
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)
        if bboxes is not None:
            draw_coco_bboxes(ax, bboxes)

        plt.show()
