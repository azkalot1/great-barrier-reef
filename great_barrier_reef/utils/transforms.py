import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomGamma(gamma_limit=(95, 105), p=1.0),
                    A.RandomToneCurve(p=1.0, scale=0.25),
                    A.RGBShift(
                        p=1.0,
                        b_shift_limit=[-70, 5],
                        r_shift_limit=[-10, 10],
                        g_shift_limit=[-10, 10],
                    ),
                    A.CLAHE(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=1.0,
            ),
            A.PixelDropout(p=0.5, per_channel=False, dropout_prob=0.1),
            A.RandomScale(scale_limit=0.35, p=1.0),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_train_transforms_crop(target_img_size=512):
    return A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(
                height=target_img_size, width=target_img_size, p=1
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms_crop(target_img_size=512):
    return A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(
                height=target_img_size, width=target_img_size, p=1
            ),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
