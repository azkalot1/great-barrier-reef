#!/bin/bash
for FOLD in 0
do
    for MODEL in tf_efficientdet_lite1 tf_efficientdet_d0 cspresdet50 cspresdext50pan
    do
        python3 train.py \
         experiment_name=$MODEL\_superheavy_fold$FOLD \
         data.transforms=train_transforms_super_heavy \
         data.augmentator_args.apply_rotation=True \
         data.augmentator_args.min_insert_starfish=7 \
         data.mosaic_augmentation=True \
         model_name=$MODEL \
         trainer_args.gpus=7 \
         data.fold=$FOLD \
         data.augmentator_args.saved_crops_path=data/crops_val_fold$FOLD.pickle
    done
done