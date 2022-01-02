import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def train_transforms_super_heavy(target_img_size=512):
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
                    A.RGBShift(
                        p=1.0,
                        b_shift_limit=[-90, 20],
                        r_shift_limit=[-20, 20],
                        g_shift_limit=[-20, 20],
                    ),
                    A.CLAHE(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomToneCurve(p=1.0, scale=0.5),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(p=0.25),
                    A.MedianBlur(p=0.25),
                    A.ToGray(p=0.25),
                ],
                p=0.1,
            ),
            A.RandomScale(scale_limit=0.25, p=1.0),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def train_transforms_heavy(target_img_size=512):
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
                    A.RGBShift(
                        p=1.0,
                        b_shift_limit=[-90, 20],
                        r_shift_limit=[-20, 20],
                        g_shift_limit=[-20, 20],
                    ),
                    A.CLAHE(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomToneCurve(p=1.0, scale=0.5),
                ],
                p=0.5,
            ),
            A.RandomScale(scale_limit=0.25, p=1.0),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def train_transforms_super_med1(target_img_size=512):
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
                    A.RandomBrightnessContrast(p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomToneCurve(p=1.0, scale=0.5),
                ],
                p=0.5,
            ),
            A.RandomScale(scale_limit=0.25, p=1.0),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def train_transforms_super_med2(target_img_size=512):
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
                    A.RGBShift(
                        p=1.0,
                        b_shift_limit=[-90, 20],
                        r_shift_limit=[-20, 20],
                        g_shift_limit=[-20, 20],
                    ),
                    A.CLAHE(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ],
                p=0.5,
            ),
            A.RandomScale(scale_limit=0.25, p=1.0),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def train_transforms_simple(target_img_size=512):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ],
                p=0.5,
            ),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def valid_transforms(target_img_size=512):
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
