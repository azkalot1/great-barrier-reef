# source: https://www.kaggle.com/bamps53/competition-metric-implementation
from ast import literal_eval
import numpy as np
from tqdm.auto import tqdm
from great_barrier_reef.dataset import StarfishDatasetAdapter
from objdetecteval.data.bbox_formats import convert_pascal_bbox_to_coco

def calc_iou(bboxes1, bboxes2, bbox_mode="xywh"):
    assert len(bboxes1.shape) == 2 and bboxes1.shape[1] == 4
    assert len(bboxes2.shape) == 2 and bboxes2.shape[1] == 4

    bboxes1 = bboxes1.copy()
    bboxes2 = bboxes2.copy()

    if bbox_mode == "xywh":
        bboxes1[:, 2:] += bboxes1[:, :2]
        bboxes2[:, 2:] += bboxes2[:, :2]

    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def f_beta(tp, fp, fn, beta=2):
    return (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp)


def calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th, verbose=False):
    gt_bboxes = gt_bboxes.copy()
    pred_bboxes = pred_bboxes.copy()

    tp = 0
    fp = 0
    for k, pred_bbox in enumerate(pred_bboxes):  # fixed in ver.7
        ious = calc_iou(gt_bboxes, pred_bbox[None, 1:])
        max_iou = ious.max()
        if max_iou > iou_th:
            tp += 1
            gt_bboxes = np.delete(gt_bboxes, ious.argmax(), axis=0)
        else:
            fp += 1
        if len(gt_bboxes) == 0:
            fp += len(pred_bboxes) - (k + 1)  # fix in ver.7
            break

    fn = len(gt_bboxes)
    return tp, fp, fn


def calc_is_correct(gt_bboxes, pred_bboxes):
    """
    gt_bboxes: (N, 4) np.array in xywh format
    pred_bboxes: (N, 5) np.array in conf+xywh format
    """
    if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, 0
        return tps, fps, fns

    elif len(gt_bboxes) == 0:
        tps, fps, fns = 0, len(pred_bboxes), 0
        return tps, fps, fns

    elif len(pred_bboxes) == 0:
        tps, fps, fns = 0, 0, len(gt_bboxes)
        return tps, fps, fns

    pred_bboxes = pred_bboxes[pred_bboxes[:, 0].argsort()[::-1]]  # sort by conf

    tps, fps, fns = 0, 0, 0
    for iou_th in np.arange(0.3, 0.85, 0.05):
        tp, fp, fn = calc_is_correct_at_iou_th(gt_bboxes, pred_bboxes, iou_th)
        tps += tp
        fps += fp
        fns += fn
    return tps, fps, fns


def calc_f2_score(gt_bboxes_list, pred_bboxes_list, verbose=False):
    """
    gt_bboxes_list: list of (N, 4) np.array in xywh format
    pred_bboxes_list: list of (N, 5) np.array in conf+xywh format
    """
    tps, fps, fns = 0, 0, 0
    for gt_bboxes, pred_bboxes in zip(gt_bboxes_list, pred_bboxes_list):
        tp, fp, fn = calc_is_correct(gt_bboxes, pred_bboxes)
        tps += tp
        fps += fp
        fns += fn
        if verbose:
            num_gt = len(gt_bboxes)
            num_pred = len(pred_bboxes)
            print(
                f"num_gt:{num_gt:<3} num_pred:{num_pred:<3} tp:{tp:<3} fp:{fp:<3} fn:{fn:<3}"
            )
    return f_beta(tps, fps, fns, beta=2)


def decode_annotations(annotaitons_str):
    """decode annotations in string to list of dict."""
    return literal_eval(annotaitons_str)


def generate_gt_from_annotation(annotations_str):
    annotations = decode_annotations(annotations_str)
    gt_bboxes = []

    for ann in annotations:
        gt_bboxes.append(np.array([ann["x"], ann["y"], ann["width"], ann["height"]]))

    gt_bboxes = np.array(gt_bboxes)
    return gt_bboxes


def prepare_gt(validation_df):
    gt_bboxes_list = []
    for ann_str in validation_df["annotations"]:
        gt_bboxes = generate_gt_from_annotation(ann_str)
        gt_bboxes_list.append(gt_bboxes)
    return gt_bboxes_list


def generate_validation_predictions(model, validation_df):
    val_predictions = []
    adapter_dataset_val = StarfishDatasetAdapter(validation_df)
    for idx in tqdm(range(len(adapter_dataset_val))):
        image, bboxes, class_labels, index, image_is_empty = adapter_dataset_val[idx]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = model.predict([image])
        # to do: make batch
        predicted_bboxes = predicted_bboxes[0]
        predicted_class_confidences = predicted_class_confidences[0]
        predictions = []
        for i in range(len(predicted_bboxes)):
            bbox = predicted_bboxes[i]
            score = predicted_class_confidences[i]
            x_min, y_min, bbox_width, bbox_height = convert_pascal_bbox_to_coco(*bbox)
            predictions.append(np.array([score, x_min, y_min, bbox_width, bbox_height]))
        val_predictions.append(np.array(predictions))
    return val_predictions


def validate_model(model, validation_df):
    predicted_boxes = generate_validation_predictions(model, validation_df)
    gt_bboxes = prepare_gt(validation_df)
    f2_score = calc_f2_score(gt_bboxes, predicted_boxes, verbose=False)
    return f2_score
