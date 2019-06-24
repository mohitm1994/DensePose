import trail
import numpy as np
import cv2
import trail
im = cv2.imread('drive/My Drive/Birthday/IMG-20180510-WA0001.jpg')
print(im)
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
cfgs = 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml'
weights = 'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
i = trail.main(im,cfgs,weights)
print(i)