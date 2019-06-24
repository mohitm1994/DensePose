# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time


from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
cfgs = 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml'
weights = 'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'

def main(im,cfgs,weights):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(cfgs)
    cfg.NUM_GPUS = 1
    weights = cache_url(weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(weights)
    print('important',model)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    im_name = 'DensePoseData/demo_data/xyz.jpg'
    output_dir = 'DensePoseData/'
    img = vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        im_name,
        output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        cls_bodys,
        dataset=dummy_coco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
    )
    return img

   
      


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args,cfgs,weights)
