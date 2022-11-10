from torch.autograd import Function
import torch.nn as nn

import torch
from torch.autograd import Variable
from torch.autograd import Variable
import numpy as np
import cv2

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1,bias=False)
        self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x,need_backprop):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.Conv2(x)
        label=self.LabelResizeLayer(x,need_backprop)
        return x,label


class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """
    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()


    def forward(self,x,need_backprop):

        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)
        for i in range(lbs.shape[0]):
            lb=np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3] ,feats.shape[2]),  interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y=Variable(torch.from_numpy(gt_blob)).cuda()
        y=y.squeeze(1).long()
        return y
