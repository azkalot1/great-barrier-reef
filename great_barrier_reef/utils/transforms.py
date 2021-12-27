import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ],
                p=0.75,
            ),
            A.RandomBrightnessContrast(p=0.2),
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


def get_train_transforms_pad(target_img_size=512):
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ],
                p=0.75,
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.PadIfNeeded(
                min_height=target_img_size,
                min_width=target_img_size,
                p=1,
                border_mode=0,
            ),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms_pad(target_img_size=1280):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=target_img_size,
                min_width=target_img_size,
                p=1,
                border_mode=0,
            ),
            A.Normalize(p=1.0),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )
