# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .DA_coco import build_city2foggy_cocostyle, 
from .DA_coco import build_city2foggy_cocostyle_source, 

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def DA_build_dataset(image_set, args):
    ## source only datasets
    if args.dataset_file == 'city2foggy_source':
        return build_city2foggy_cocostyle_source(image_set, args)

    ## DA datasets
    if args.dataset_file == 'city2foggy':
        return build_city2foggy_cocostyle(image_set, args)
   
    raise ValueError(f'dataset {args.dataset_file} not supported')

