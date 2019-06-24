import trail
import numpy as np
import cv2
import trail
from caffe2.python import workspace
im = cv2.imread('DensePoseData/demo_data/xyz.jpg')
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
cfgs = 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml'
weights = 'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
i = trail.main(im,cfgs,weights)
cv2.imwrite('DensePoseData/x_IUV.png',i)
