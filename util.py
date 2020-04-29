from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

def unique(tensor):
    # 去除其中重复的元素，并按元素由小到大返回一个新的无元素重复的元组或者列表
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(bbox1, bbox2):
    # 方框左上角和右下角位置坐标
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:,0], bbox1[:,1], bbox1[:,2], bbox1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:,0], bbox2[:,1], bbox2[:,2], bbox2[:,3]
    # 计算交集面积
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min=0)*torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)

    b1_area = (b1_x2-b1_x1+1)*(b1_y2-b1_y1+1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2-b2_y1+1)
    
    iou = inter_area / (b1_area+b2_area-inter_area)
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    prediction[:,:,:4] *= stride

    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # 去掉置信度低的结果
    conf_mask = (prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    # 计算锚框的大小和中心位置
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0]-prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1]-prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0]+prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1]+prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        # 计算每个锚框预测概率最大的结果
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        img_classes = unique(image_pred_[:,-1])

        for cls_ in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1]==cls_).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            for i in range(idx):
                # 计算每个框的IoU
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break
                # 去除交并比高并且置信度低的框
                iou_mask = (ious<nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    
    try:
        return output
    except:
        return 0

def letterbox_image(img, inpdim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inpdim
    new_w = int(img_w*min(w/img_w, h/img_h))
    new_h = int(img_h*min(w/img_w, h/img_h))
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inpdim[1], inpdim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2+new_h, (w-new_w)//2:(w-new_w)//2+new_w, :] = resized_img

    return canvas
    
def prep_image(img, inp_dim):

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    orig_im = cv2.resize(orig_im, (416, 416))
    img_ = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img_[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0).cuda()
    return img_, orig_im, dim    
