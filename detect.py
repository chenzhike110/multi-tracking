from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from util import *
from read_build import Darknet
import argparse
import pickle as pkl 
import pandas as pd 
import random
import cv2 

def arg_parse():
    
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest='images', help='检测图像', default='C:\\Users\\86177\\Desktop\\SharedScreenshot.jpg', type=str)
    parser.add_argument("--batchsize", dest='batch', help='批量数', default= 1)
    parser.add_argument("--confidence", dest='confidence', help='检测目标置信度', default=0.5)
    parser.add_argument("--nms_thresh", dest='thresh', help='合并检测框的交并比', default=0.1)
    parser.add_argument("--cfg", dest='cfgfile', help='配置文件', default='yolo.cfg', type=str)
    parser.add_argument("--weights", dest='weightsfile', help='权重文件', default='E:\\SRTP\\deep_sort_pytorch-master\\detector\\YOLOv3\\weight\\yolov3.weights', type=str)
    parser.add_argument("--reso", dest='reso', help='网络大小（越大越精确，越小越快)', default=416, type=int)

    return parser.parse_args()

def load_classes(namesfile):
    fp = open(namesfile,'r')
    names = fp.read().split('\n')[:-1]
    return names

def write(x, img, color):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls_ = int(x[-1])
    label = "{0}".format(classes[cls_])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,1,1)[0]
    c2 = c1[0]+t_size[0]+3, c1[1]+t_size[1]+4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 1)
    return img


args = arg_parse()
images = args.images
# batch_size = args.batch
confidence = args.confidence
nms_thresh = args.thresh
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("coco.names")

print("Loading network...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("model already...")

model.net_info["height"] = args.reso
inp_dim = args.reso
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

img_, orig_im, dim = prep_image(args.images, 416)
with torch.no_grad():
    prediction = model(Variable(img_, CUDA))
prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

for i in range(len(prediction)):
    img = write(prediction[i], orig_im, [255,255,255])
cv2.imshow("result", img)
cv2.waitKey(0)