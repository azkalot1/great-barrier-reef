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
