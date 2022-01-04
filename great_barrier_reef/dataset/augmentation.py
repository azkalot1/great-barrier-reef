import pandas as pd
from ast import literal_eval
from .utils import get_bboxes_from_annotation
import cv2
import PIL
import numpy as np
import random
from tqdm.auto import tqdm
from skimage.exposure import match_histograms
import pickle


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std
    (sigma)"""
    target = np.exp(-(((pos - muy) / sigma) ** 2) / 2)
    return target


def random_rotate(image):
    flag = np.random.choice([-1, 0, 1, 2])
    if flag == -1:
        return image
    else:
        return cv2.rotate(image, flag)


class ImageInsertAug(object):
    def __init__(
        self,
        non_empty_df=None,
        images_dir_path="../data/train_images/",
        min_insert_starfish=1,
        max_insert_starfish=11,
        lambda_insert=0.3,
        blue_thr=200,
        max_attempts_insert=3,
        saved_crops_path=None,
        apply_rotation=False,
        match_histograms=False,
    ):
        self.images_dir_path = images_dir_path
        self.image_paths = non_empty_df.apply(
            lambda x: "video_{}/{}.jpg".format(x["video_id"], x["video_frame"]), axis=1
        ).values
        if saved_crops_path is not None:
            with open(saved_crops_path, "rb") as input_file:
                self.starfish_crops = pickle.load(input_file)
        else:
            self.annotations = non_empty_df["annotations"].apply(literal_eval).values
            self.starfish_crops = self.prepare_crops()
        self.min_insert_starfish = min_insert_starfish
        self.max_insert_starfish = max_insert_starfish

        self.lambda_insert = lambda_insert
        self.blue_thr = blue_thr
        self.max_attempts_insert = max_attempts_insert
        self.apply_rotation = apply_rotation
        self.match_histograms = match_histograms

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
            if self.apply_rotation:
                selected_starfish = random_rotate(selected_starfish)
            h, w = selected_starfish.shape[:2]
            if self.match_histograms:
                selected_starfish = match_histograms(
                    selected_starfish, image_inserted, multichannel=True
                )
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


class MosaicAugmentator(object):
    def __init__(self, low_s=0.1, high_s=0.85, min_size_x=10, min_size_y=10):
        self.low_s = low_s
        self.high_s = high_s
        self.min_size_x = min_size_x
        self.min_size_y = min_size_y

    def run_augmentation(self, images, bboxes):
        aug_img = np.zeros_like(images[0])
        aug_bboxes = []

        size = images[1].shape[:2]
        yp, xp = [
            int(random.uniform(size[i] * self.low_s, size[i] * self.high_s))
            for i in range(2)
        ]
        for i in range(4):
            if i == 0:  # top left corner
                miny, minx, maxy, maxx = 0, 0, yp, xp
                aug_img[:yp, :xp, :] = images[i][:yp, :xp, :]

            elif i == 1:  # top right
                miny, minx, maxy, maxx = 0, xp, yp, size[1]
                aug_img[:yp, xp:, :] = images[i][:yp, xp:, :]

            elif i == 2:  # bottom left
                miny, minx, maxy, maxx = yp, 0, size[0], xp
                aug_img[yp:, :xp, :] = images[i][yp:, :xp, :]

            elif i == 3:  # bottom right
                miny, minx, maxy, maxx = yp, xp, size[0], size[1]
                aug_img[yp:, xp:, :] = images[i][yp:, xp:, :]

            img_bboxes = bboxes[i]
            if len(img_bboxes) > 0:
                mask = (
                    (img_bboxes[:, 1] <= maxy)
                    & (img_bboxes[:, 3] >= miny)
                    & (img_bboxes[:, 0] <= maxx)
                    & (img_bboxes[:, 2] >= minx)
                )
                img_bboxes = img_bboxes[mask, :]
                if len(img_bboxes) > 0:
                    img_bboxes[:, 1] = np.clip(img_bboxes[:, 1], miny + 1, maxy - 1)
                    img_bboxes[:, 3] = np.clip(img_bboxes[:, 3], miny + 1, maxy - 1)
                    img_bboxes[:, 0] = np.clip(img_bboxes[:, 0], minx + 1, maxx - 1)
                    img_bboxes[:, 2] = np.clip(img_bboxes[:, 2], minx + 1, maxx - 1)
                    aug_bboxes.append(img_bboxes)

        if len(aug_bboxes) > 0:
            aug_bboxes = np.concatenate(aug_bboxes)

            h = aug_bboxes[:, 3] - aug_bboxes[:, 1]
            w = aug_bboxes[:, 2] - aug_bboxes[:, 0]

            mask_size = (h >= self.min_size_y) & (w >= self.min_size_x)
            aug_bboxes = aug_bboxes[mask_size, :]  # can go to zero here

            labels = np.ones(len(aug_bboxes))
            image_is_empty = False

        if len(aug_bboxes) == 0:
            # wow, no bboxes!
            aug_bboxes = np.array([[0, 0, 1, 1]])
            labels = np.zeros(1)
            image_is_empty = True

        return PIL.Image.fromarray(aug_img), aug_bboxes, labels, image_is_empty
