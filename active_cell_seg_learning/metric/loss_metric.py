import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def accuracy(x, y):
    """
    Compute Accuracy of 2 Pytorch Tensors.
    """
    compare = (x.float() == y.float())
    acc = compare.sum().item()

    return acc/len(x.flatten())

def focal_loss_metric(pred, targets, alpha=0.8, gamma=2, smooth=1, classes = 2):
    if classes == 2:
        ### Using BC for Cases with 2 Classes
        pred = pred.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(pred, targets, reduction='mean')
    else:
        ### BCE for >2 Classes
        BCE = F.cross_entropy(input=pred, target=targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
    return focal_loss

def IoU_loss_metric(pred, targets, smooth=1):
    #flatten label and prediction tensors
    inputs = pred.view(-1)
    targets = targets.view(-1)
        
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
    return IoU

def iou_mean(pred, targets, n_classes=3):

    ious = []
    pred = pred.view(-1)
    target = targets.view(-1)

    count = np.zeros(n_classes)

    for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().cpu().item()
        union = pred_inds.long().sum().cpu().item() + target_inds.long().sum().cpu().item() - intersection  # .data.cpu()[0] - intersection

        if union == 0:
            ious.append(0.0)
        else:
            count[cls] += 1
            ious.append(float(intersection) / float(max(union, 1)))
    iou_mean = np.divide(np.sum(ious), n_classes) 
    return iou_mean