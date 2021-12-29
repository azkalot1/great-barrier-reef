import pandas as pd
from ast import literal_eval
from .utils import get_bboxes_from_annotation
import PIL
import numpy as np
from tqdm.auto import tqdm


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std
    (sigma)"""
    target = np.exp(-(((pos - muy) / sigma) ** 2) / 2)
    return target


class ImageInsertAug(object):
    def __init__(
        self,
        non_empty_df,
        images_dir_path="../data/train_images/",
        min_insert_starfish=3,
        max_insert_starfish=7,
        lambda_insert=0.3,
        blue_thr=220,
        max_attempts_insert=3,
    ):
        self.images_dir_path = images_dir_path
        self.image_paths = non_empty_df.apply(
            lambda x: "video_{}/{}.jpg".format(x["video_id"], x["video_frame"]), axis=1
        ).values
        self.annotations = non_empty_df["annotations"].apply(literal_eval).values
        self.min_insert_starfish = min_insert_starfish
        self.max_insert_starfish = max_insert_starfish
        self.starfish_crops = self.prepare_crops()
        self.lambda_insert = lambda_insert
        self.blue_thr = blue_thr
        self.max_attempts_insert = max_attempts_insert

    def prepare_crops(self):
        starfish_crops = []
        for idx in tqdm(range(len(self.annotations))):
            bboxes_image, _ = get_bboxes_from_annotation(
                self.annotations[idx], 1280, 720
            )
            starfish_crops.extend(
                self.extract_from_image(bboxes_image, self.image_paths[idx])
            )
        return starfish_crops

    def extract_from_image(self, bboxes, image_path):
        image = np.array(PIL.Image.open(self.images_dir_path + "/" + image_path))
        starfish_crops = []
        for bbox in bboxes:
            starfish_crops.append(image[bbox[1] : bbox[3], bbox[0] : bbox[2]])
        return starfish_crops

    def insert_starfish(self, image_inserted, selected_starfish, y_ins, x_ins, h, w):
        inserting_region = image_inserted[y_ins : (y_ins + h), x_ins : (x_ins + w)]
        image_inserted[y_ins : (y_ins + h), x_ins : (x_ins + w)] = (
            self.lambda_insert * inserting_region
            + (1 - self.lambda_insert) * selected_starfish
        )
        return image_inserted

    def update_insertion_matrix(self, insertion_matrix, y_ins, x_ins, h, w):
        insertion_matrix[y_ins : (y_ins + h), x_ins : (x_ins + w)] = 1
        return insertion_matrix

    def augment_image(self, image):
        image_inserted = image.copy()
        insertion_matrix = np.zeros(image.shape[:2])

        selected_number_of_insertions = np.random.choice(
            range(self.min_insert_starfish, self.max_insert_starfish)
        )
        bboxes = []
        for _ in range(selected_number_of_insertions):
            starfish_idx = np.random.choice(range(len(self.starfish_crops)))
            selected_starfish = self.starfish_crops[starfish_idx]
            h, w = selected_starfish.shape[:2]
            allow_to_insert = False
            insertion_attempt = 0
            while not allow_to_insert and insertion_attempt <= self.max_attempts_insert:
                y_ins, x_ins = (
                    np.random.choice(range(h + 1, 720 - h)),
                    np.random.choice(range(w + 1, 1280 - w)),
                )
                is_original = np.allclose(
                    insertion_matrix[y_ins : (y_ins + h), x_ins : (x_ins + w)].sum(), 0
                )
                is_blue = (
                    image[y_ins : (y_ins + h), x_ins : (x_ins + w)][..., 2].mean()
                    >= self.blue_thr
                )
                if is_original and not is_blue:
                    allow_to_insert = True
                    image_inserted = self.insert_starfish(
                        image_inserted, selected_starfish, y_ins, x_ins, h, w
                    )
                    insertion_matrix = self.update_insertion_matrix(
                        insertion_matrix, y_ins, x_ins, h, w
                    )
                    bboxes.append([x_ins, y_ins, x_ins + w, y_ins + w])
                else:
                    insertion_attempt += 1

        return image_inserted, bboxes
