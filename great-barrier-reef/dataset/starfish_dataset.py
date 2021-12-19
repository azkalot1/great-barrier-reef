from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import PIL
from pathlib import Path
import matplotlib.pyplot as plt


class StarfishDatasetAdaptor(Dataset):
    def __init__(
        self,
        annotations_dataframe: pd.DataFrame,
        images_dir_path: str = "../data/train_images/",
    ):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.image_paths = self.prepare_image_paths()

    def prepare_image_paths(self) -> np.ndarray:
        image_paths = self.annotations_df.apply(
            lambda x: "video_{}/{}.jpg".format(x["video_id"], x["video_frame"]), axis=1
        )
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.image_paths[index]
        image = PIL.Image.open(self.images_dir_path / image_name)

        return image, image_name

    def show_image(self, index):
        """get image by its index and plot."""
        image, image_name = self.get_image_and_labels_by_idx(index)
        print(f"image_name: {image_name}")
        self._show_image(image)

    def _show_image(self, image, figsize=(10, 10)):
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)

        plt.show()
