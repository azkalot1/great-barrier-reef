import torch
import random
import numpy as np


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


def fix_bboxes(bboxes: list, im_width: int, im_height: int):
    bboxes_fixed = []
    for bbox in bboxes:
        bboxes_fixed.append(
            [
                bbox[0],
                bbox[1],
                min(bbox[2], im_width),
                min(bbox[3], im_height),
            ]
        )
    return bboxes_fixed


def get_area(annotation):
    total_bbox_area_images = 0
    for ann in annotation:
        total_bbox_area_images += ann["width"] * ann["height"]
    return total_bbox_area_images


class MosaicAugmentator(object):
    def __init__(self, low_s=0.1, high_s=0.85):
        self.low_s = low_s
        self.high_s = high_s

    def run_augmentation(self, images, bboxes):
        plot_img = torch.zeros_like(images[0])
        updated_bboxes = []

        size = images[1].shape[1:]
        yp, xp = [
            int(random.uniform(size[i] * self.low_s, size[i] * self.high_s))
            for i in range(2)
        ]

        for i in range(4):
            if i == 0:  # top left corner
                miny, minx, maxy, maxx = 0, 0, yp, xp
                plot_img[:, :yp, :xp] = images[i][:, :yp, :xp]

            elif i == 1:  # top right
                miny, minx, maxy, maxx = 0, xp, yp, size[1]
                plot_img[:, :yp, xp:] = images[i][:, :yp, xp:]

            elif i == 2:  # bottom left
                miny, minx, maxy, maxx = yp, 0, size[0], xp
                plot_img[:, yp:, :xp] = images[i][:, yp:, :xp]

            elif i == 3:  # bottom right
                miny, minx, maxy, maxx = yp, xp, size[0], size[1]
                plot_img[:, yp:, xp:] = images[i][:, yp:, xp:]

            img_bboxes = bboxes[i]
            mask = (
                (img_bboxes[:, 0] <= maxy)
                & (img_bboxes[:, 2] >= miny)
                & (img_bboxes[:, 1] <= maxx)
                & (img_bboxes[:, 3] >= minx)
            )
            img_bboxes = img_bboxes[mask, :]
            if len(bboxes) > 0:
                img_bboxes[:, 0] = torch.clip(img_bboxes[:, 0], miny + 1, maxy - 1)
                img_bboxes[:, 2] = torch.clip(img_bboxes[:, 2], miny + 1, maxy - 1)
                img_bboxes[:, 1] = torch.clip(img_bboxes[:, 1], minx + 1, maxx - 1)
                img_bboxes[:, 3] = torch.clip(img_bboxes[:, 3], minx + 1, maxx - 1)
                updated_bboxes.append(img_bboxes)

        updated_bboxes = torch.concat(updated_bboxes)
        labels = torch.from_numpy(np.ones(len(updated_bboxes)))
        if len(updated_bboxes) == 0:
            # wow, no bboxes!
            updated_bboxes = np.array([[0, 0, 1, 1]])
            labels = np.zeros(1)
        return plot_img, updated_bboxes, labels

    def augment_batch(self, images, bboxes):
        bs = images.size()[0]
        augment_images = []
        augment_bboxes = []
        augment_labels = []
        for _ in range(bs):
            idx = np.random.choice(range(bs), size=4, replace=True)
            updated_img, updated_bboxes, updated_labels = self.run_augmentation(
                images[idx], [bboxes[i] for i in idx]
            )
            augment_images.append(updated_img)
            augment_bboxes.append(updated_bboxes)
            augment_labels.append(updated_labels)

        augment_images = torch.stack(augment_images)
        return augment_images, augment_bboxes, augment_labels
