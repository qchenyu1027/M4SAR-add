# YOLOv5 common modules

import math
from collections import OrderedDict
from copy import copy
from functools import partial

from pathlib import Path
import warnings
import warnings
from typing import Callable, Any,Optional

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from fvcore.nn import parameter_count, flop_count
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
from torch.utils import checkpoint

# from utils.datasets import letterbox
# from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mmcv.ops import DeformConv2dPack as DCN
from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import einops
try:
    from timm.models.layers import trunc_normal_, LayerNorm2d
except ImportError:
    print("[Info] timm not found, using mock implementation.")
    def trunc_normal_(tensor, std=0.02, mean=0.0, a=-2.0, b=2.0):
        nn.init.normal_(tensor, mean=mean, std=std)
        # 简单截断
        with torch.no_grad():
            tensor.clamp_(a, b)
            
    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x):
            x = x.permute(0, 2, 3, 1)
            x = super().forward(x)
            return x.permute(0, 3, 1, 2)

# 模拟 natten 的部分功能
try:
    from natten.functional import na2d_qk, na2d_av
except ImportError:
    print("[Info] natten not found, using mock implementation for shape testing.")
    
    def na2d_qk(q, k, kernel_size, dilation, is_causal, rpb=None):
        # q, k shape: [B, Heads, H, W, Dim]
        # output attn shape: [B, Heads, H, W, K*K]
        B, Heads, H, W, Dim = q.shape
        K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        return torch.randn(B, Heads, H, W, K*K, device=q.device)

    def na2d_av(attn, v, kernel_size, dilation, is_causal):
        # attn shape: [B, Heads, H, W, K*K]
        # v shape: [B, Heads, H, W, Dim]
        # output shape: [B, Heads, H, W, Dim]
        B, Heads, H, W, _ = attn.shape
        Dim = v.shape[-1]
        return torch.randn(B, Heads, H, W, Dim, device=v.device)
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class ConvAcceptReturn(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1,mode=0, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvAcceptReturn, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.mode=mode

    # def forward(self, x):
    #     return self.act(self.bn(self.conv(x)))

    def forward(self, x):
        x=x[0]+x[1][self.mode]
        return self.act(self.conv(x))

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class VGGblock(nn.Module):
    def __init__(self, num_convs, c1, c2):
        super(VGGblock, self).__init__()
        self.blk = []
        for num in range(num_convs):
            if num == 0:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
            else:
                self.blk.append(nn.Sequential(nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, padding=1),
                                              nn.ReLU(),
                                              ))
        self.blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vggblock = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.vggblock(x)

        return out


class ResNetblock(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1):
        super(ResNetblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=self.expansion*c2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*c2)

        self.shortcut = nn.Sequential()
        if stride != 1 or c1 != self.expansion*c2:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=self.expansion*c2, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion*c2),
                                          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetlayer(nn.Module):
    expansion = 4

    def __init__(self, c1, c2, stride=1, is_first=False, num_blocks=1):
        super(ResNetlayer, self).__init__()
        self.blk = []
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=7, stride=2, padding=3, bias=False),
                                        nn.BatchNorm2d(c2),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.blk.append(ResNetblock(c1, c2, stride))
            for i in range(num_blocks - 1):
                self.blk.append(ResNetblock(self.expansion*c2, c2, 1))
            self.layer = nn.Sequential(*self.blk)

    def forward(self, x):
        out = self.layer(x)

        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class Add(nn.Module):
    # Add a list of tensors and averge
    def __init__(self, weight=0.5):
        super().__init__()
        self.w = weight

    def forward(self, x):
        return x[0] * self.w + x[1] * (1 - self.w)




class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])

class Add3(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c2):
        super().__init__()

    def forward(self, x):
        y = x[0] + x[1]
        return y

class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.concat(x)
        x = self.act(self.conv(x))

        return x


class DMAF(nn.Module):
    def __init__(self, c2):
        super(DMAF, self).__init__()


    def forward(self, x):
        x1 = x[0]
        x2 = x[1]

        subtract_vis = x1 - x2
        avgpool_vis = nn.AvgPool2d(kernel_size=(subtract_vis.size(2), subtract_vis.size(3)))
        weight_vis = torch.tanh(avgpool_vis(subtract_vis))

        subtract_ir = x2 - x1
        avgpool_ir = nn.AvgPool2d(kernel_size=(subtract_ir.size(2), subtract_ir.size(3)))
        weight_ir = torch.tanh(avgpool_ir(subtract_ir))

        x1_weight = subtract_vis * weight_ir
        x2_weight = subtract_ir * weight_vis

        return x1_weight, x2_weight


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att_vis = torch.matmul(q_ir, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_vis, k_ir) / np.sqrt(self.d_k)
        # att_vis = torch.matmul(k_vis, q_ir) / np.sqrt(self.d_k)
        # att_ir = torch.matmul(k_ir, q_vis) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)

        # output
        out_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]


class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))

            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]

class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(*[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        # VSS_layer==========open=============
        # rgb_fea=rgb_fea.permute(0,3,1,2).contiguous()
        # ir_fea=ir_fea.permute(0,3,1,2).contiguous()
        # print(rgb_fea.shape)
        # print(ir_fea.shape)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#
        
        return new_fea


class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# 通道注意力模块
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        '''
        :param in_channels: 输入通道数
        :param reduction_ratio: 输出通道数量的缩放系数
        :param pool_types: 池化类型
        '''

        super(Channel_Attention, self).__init__()

        self.pool_types = pool_types
        self.in_channels = in_channels
        self.shared_mlp = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=in_channels, out_features=in_channels//reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(in_features=in_channels//reduction_ratio, out_features=in_channels)
                                        )

    def forward(self, x):
        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        output = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * output


# 空间注意力模块
class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spatial_Attention, self).__init__()

        self.spatial_attention = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
                                               nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
                                               )

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)  # 在通道维度上分别计算平均值和最大值，并在通道维度上进行拼接
        x_output = self.spatial_attention(x_compress)  # 使用7x7卷积核进行卷积
        scaled = nn.Sigmoid()(x_output)

        return x * scaled  # 将输入F'和通道注意力模块的输出Ms相乘，得到F''


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        super(CBAM, self).__init__()

        self.spatial = spatial
        self.channel_attention = Channel_Attention(in_channels=in_channels, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)

    def forward(self, x):
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from models.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
try:
    from selective_scan.selective_scan.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
except:
    pass

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda


    class SelectiveScan(torch.autograd.Function):
        # @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True
            if not u.is_cuda:
                return u
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        # @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)


    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None


    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B,  d, 2 * H * W
            # ys = ys[:, 0] + ys[:, 1]  # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H * W].view(B, -1, H, W), ys[:, :, H * W:2 * H * W].view(B, -1, H, W)


    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L // 2], ys[:, :, L // 2:L]

        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2 * L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2 * L)
            return xs, None, None


    def cross_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)
        y=y.half()
        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def selective_scan_1d(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape  # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def cross_selective_scan_multimodal_k1(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W)

        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L // 2] + ys[:, 0, :, L // 2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)

        return y


    def cross_selective_scan_multimodal_k2(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm1: torch.nn.Module = None,
            out_norm2: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e)  # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        # @staticmethod
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W).to(x_rgb.dtype)

        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)

        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[1].debugName().startswith("dts")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    assert inputs[4].debugName().startswith("Cs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


DEV = False


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 4 if not (self.forward_core == self.forward_corev1_share_ssm) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K = 1
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K if not (self.forward_core == self.forward_corev1_share_a) else 1
        # VMamaba set K=4, while original SSM set K=1
        if self.forward_core == self.forward_corev0:
            self.K2 = 1
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 1  # 4

        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,  # z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            # return_last_state=False,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        # inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = torch.transpose(out_y[:, 0], dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)

        return y

    def forward_corev1(self, x: torch.Tensor, float32=True):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        As = -torch.exp(self.A_logs.to(torch.float))  # (k * d, d_state)
        Ds = self.Ds.to(torch.float)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.to(torch.float).view(-1)  # (k * d)

        if float32:
            ys: torch.Tensor = selective_scan(
                xs.to(torch.float),
                dts.to(torch.float),
                As,
                Bs.to(torch.float),
                Cs.to(torch.float),
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        else:
            out_y: torch.Tensor = selective_scan(
                xs, dts,
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16

            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

            # if torch.isinf(y).any() or torch.isnan(y).any():
            #     for item in [y, xs, dts, As, Bs, Cs, Ds]:
            #         print(torch.isinf(item).any(), torch.isnan(item).any(), item.max(), item.min())
            #     import time; time.sleep(10000)

        return y

    def forward_corev1_share_ssm(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)
            return xs

        x_dbl = torch.einsum("b d l, c d -> b c l", x.view(B, -1, L), self.x_proj_weight[0])
        # x_dbl = x_dbl + self.x_proj_bias.view(1, -1, 1)
        dt, BC = torch.split(x_dbl, [self.dt_rank, 2 * self.d_state], dim=1)
        dt = torch.einsum("b r l, d r -> b d l", dt, self.dt_projs_weight[0])
        x_dt_BC = torch.cat([x, dt.view(B, -1, H, W), BC.view(B, -1, H, W)], dim=1)  # (b, -1, h, w)

        x_dt_BCs = cross_scan_2d(x_dt_BC)  # (b, k, d, l)
        xs, dts, Bs, Cs = torch.split(x_dt_BCs, [self.d_inner, self.d_inner, self.d_state, self.d_state], dim=2)

        xs = xs.contiguous().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        As = -torch.exp(self.A_logs.float()).repeat(4, 1)  # (k * d, d_state)
        Ds = self.Ds.repeat(4)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1).repeat(4)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 4, -1, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1_share_a(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x, dim=1):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=dim)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=dim)  # (b, k, d, l)
            return xs

        K = 4
        xs = cross_scan_2d(x, dim=1)  # (b, d, k, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.to(xs.dtype).view(1, K, -1, 1)

        xs = xs.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        dts = dts.transpose(dim0=1, dim1=2).contiguous().view(B, -1, K * L)
        As = -torch.exp(self.A_logs.float())  # (D, N)
        Ds = self.Ds.view(-1)  # (D)
        Bs = Bs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)
        Cs = Cs.transpose(dim0=1, dim1=2).contiguous().view(B, 1, -1, K * L)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=None,
            delta_softplus=True,
        ).view(B, -1, 4, L)
        # assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, :, 2:4], dims=[-1]).view(B, -1, 2, L)
        wh_y = torch.transpose(out_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, :, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, :, 0].float() + inv_y[:, :, 0].float() + wh_y.float() + invwh_y.float()

        if self.softmax_version:
            y = torch.softmax(y, dim=-1).to(x.dtype)
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
            nrows=nrows,
        )

    def forward_core_1d(self, x: torch.Tensor):
        return selective_scan_1d(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
        )

    # forward_core = forward_core_share_ssm
    # forward_core = forward_core_share_a
    # forward_core = forward_corev1
    forward_core = forward_corev2  # vmamba

    # forward_core = forward_corev0 # ori mamba
    # forward_core = forward_core_1d # ori mamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out



# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y



# ====================================================
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CM_Attention(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None),
            self.softmax_version,
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb= x_rgb.permute(0,2,3,1).contiguous()
        x_e =x_e.permute(0,2,3,1).contiguous()
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans))  # (b, d, h, w)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv)  # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()  # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y)).permute(0,3,1,2).contiguous()

        return out



class CMSSF(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CM_Attention(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e = x[1]
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)







class DBISSF_Attention(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.Cross = Cross
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_share = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_share = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.A_log_share = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)
        self.D_share = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)
        self.out_norm_share = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def visualization(feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D,feat_fusion,h,w):

        save_dir = 'visualization'
        feat_rgb_A = torch.mean(feat_rgb_A, dim=1)
        feat_rgb_B = torch.mean(feat_rgb_B, dim=1)
        feat_rgb_C = torch.mean(feat_rgb_C, dim=1)
        feat_rgb_D = torch.mean(feat_rgb_D, dim=1)
        feat_e_A = torch.mean(feat_e_A, dim=1)
        feat_e_B = torch.mean(feat_e_B, dim=1)
        feat_e_C = torch.mean(feat_e_C, dim=1)
        feat_e_D = torch.mean(feat_e_D, dim=1)
        block = [feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D]
        black_name = ["feat_rgb_A","feat_rgb_B","feat_rgb_C","feat_rgb_D","feat_e_A","feat_e_B","feat_e_C","feat_e_D"]
        plt.figure()
        for i in range(len(block)):
            feature = transforms.ToPILImage()(block[i].squeeze())
            ax = plt.subplot(3, 3, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(black_name[i], fontsize=8)
            plt.imshow(feature)
        plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)

    def forward(self, x_rgb: torch.Tensor,x_share_rgb, x_e: torch.Tensor,x_share_e):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_share_rgb = x_share_rgb.permute(0, 2, 1)
        x_share_e = x_share_e.permute(0, 2, 1)

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        x_dbl_share = self.x_proj_share(rearrange(torch.add(x_share_rgb,x_share_e), "b d l -> (b l) d"))  # (bl d)

        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_share, B_share, C_share = torch.split(x_dbl_share, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_share = self.dt_proj_share.weight @ dt_share.t()

        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        dt_share = rearrange(dt_share, "d (b l) -> b d l", l=L)

        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        A_share = -torch.exp(self.A_log_share.float())  # (k * d, d_state)

        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_share = rearrange(B_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_share = rearrange(C_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        if self.Cross:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_e, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_rgb, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus = True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        else:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_rgb, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_e, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        y_share_rgb = rearrange(y_share_rgb, "b d l -> b l d")
        y_share_rgb = self.out_norm_share(y_share_rgb)
        y_share_e = rearrange(y_share_e, "b d l -> b l d")
        y_share_e = self.out_norm_share(y_share_e)
        return y_rgb,y_share_rgb, y_e,y_share_e
class DBISSF_SS(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            learnableweight=False,
            scan = False,
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.Cross = Cross
        self.learnableweight=learnableweight
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_share = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if learnableweight:
            self.learnableweight_rgb = LearnableWeights()
            self.learnableweight_e = LearnableWeights()
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_share = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_share = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = DBISSF_Attention(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            Cross=Cross,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,

            **kwargs,
        )
        self.scan = scan
        if scan:
            self.CMA_ssm2 = DBISSF_Attention(
                d_model=self.d_model,
                d_state=self.d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                Cross=Cross,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,

                **kwargs,
            )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = x_rgb.permute(0, 2,3,1).contiguous()
        x_e = x_e.permute(0, 2,3,1).contiguous()
        temp_rgb = x_rgb
        temp_e =x_e
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        x_share_rgb =self.in_proj_share(temp_rgb)
        x_share_e = self.in_proj_share(temp_e)

        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb = x_rgb.permute(0, 3,1,2).contiguous()
            x_e = x_e.permute(0, 3,1,2).contiguous()
            x_share_rgb = x_share_rgb.permute(0, 3, 1, 2).contiguous()
            x_share_e = x_share_e.permute(0, 3, 1, 2).contiguous()

            x_rgb_conv = self.act(self.conv2d(x_rgb))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e))  # (b, d, h, w)
            x_share_rgb_conv = self.act(self.conv2d(x_share_rgb))  # (b, d, h, w)
            x_share_e_conv = self.act(self.conv2d(x_share_e))  # (b, d, h, w)

            x_rgb_conv_new = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv_new = rearrange(x_e_conv, "b d h w -> b (h w) d")
            x_share_rgb_conv_new = rearrange(x_share_rgb_conv, "b d h w -> b (h w) d")
            x_share_e_conv_new = rearrange(x_share_e_conv, "b d h w -> b (h w) d")

            y_rgb, y_share_rgb, y_e, y_share_e = self.CMA_ssm(x_rgb_conv_new,x_share_rgb_conv_new, x_e_conv_new, x_share_e_conv_new)
            # to b, d, h, w
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)
            y_share_rgb = y_share_rgb.view(B, H, W, -1)
            y_share_e = y_share_e.view(B, H, W, -1)
            if self.scan:
                x_rgb_conv2=x_rgb_conv
                x_e_conv2=x_e_conv
                x_share_rgb_conv2=x_share_rgb
                x_share_e_conv2=x_share_e
                x_rgb_conv_new2 = rearrange(x_rgb_conv2, "b d h w -> b (w h) d")
                x_e_conv_new2 = rearrange(x_e_conv2, "b d h w -> b (w h) d")
                x_share_rgb_conv_new2 = rearrange(x_share_rgb_conv2, "b d h w -> b (w h) d")
                x_share_e_conv_new2 = rearrange(x_share_e_conv2, "b d h w -> b (w h) d")
                y_rgb2, y_share_rgb2, y_e2, y_share_e2 = self.CMA_ssm2(x_rgb_conv_new2, x_share_rgb_conv_new2, x_e_conv_new2,
                                                                  x_share_e_conv_new2)
                y_rgb2 = y_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_e2 = y_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_rgb2 = y_share_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_e2 = y_share_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_rgb=0.5*y_rgb+0.5*y_rgb2
                y_e = 0.5*y_e+0.5*y_e2
                y_share_rgb=0.5*y_share_rgb+0.5*y_share_rgb2
                y_share_e=0.5*y_share_e+0.5*y_share_e2
        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb)).permute(0, 3,1,2).contiguous()
        out_e = self.dropout_e(self.out_proj_e(y_e)).permute(0, 3,1,2).contiguous()
        out_share_rgb = self.dropout_share(self.out_proj_share(y_share_rgb)).permute(0, 3, 1, 2).contiguous()
        out_share_e = self.dropout_share(self.out_proj_share(y_share_e)).permute(0, 3, 1, 2).contiguous()

        return out_rgb, out_share_rgb, out_e, out_share_e


class DBISSF(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            return_separately:bool = True,
            Cross:bool = True,
            scan:bool = False,
            learnableweight:bool=False,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.return_serparately = return_separately
        self.op = DBISSF_SS(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            Cross = Cross,
            learnableweight=learnableweight,
            scan=scan,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path_share = DropPath(drop_path)
        # LearnableCoefficient

        self.return_serparately=return_separately
        if not return_separately:
            self.learnable_weight = LearnableWeights()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e =x[1]
        x_rgb_cross,x_share_rgb, x_e_cross,x_share_e = self.op(x_rgb, x_e)

        b,d,h,w= x_e.shape
        x_rgb_new = x_rgb + self.drop_path1(x_rgb_cross)
        x_e_new = x_e + self.drop_path2(x_e_cross)
        x_share_rgb = x_share_rgb +self.drop_path_share(x_share_rgb)
        x_share_e = x_share_e + self.drop_path_share(x_share_e)
        if self.return_serparately:
            return x_rgb_new,x_share_rgb, x_e_new, x_share_e
        else:
            x_cross_feat = self.learnable_weight(x_rgb_new,x_e_new)
        #=================visualization==================
        # save_dir = 'visualization'
        # fea_rgb = torch.mean(x_rgb, dim=1)
        # fea_rgb_CFE = torch.mean(x_rgb_cross, dim=1)
        # fea_rgb_new = torch.mean(x_rgb_new, dim=1)
        # fea_ir = torch.mean(x_e, dim=1)
        # fea_ir_CFE = torch.mean(x_e_cross, dim=1)
        # fea_ir_new = torch.mean(x_e_new, dim=1)
        # fea_new = torch.mean(x_cross_feat, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After cross', 'fea_rgb new', 'fea_ir', 'fea_ir After cross', 'fea_ir new',
        #               'fea_ir mambafusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        #     plt.show()
        # plt.savefig(save_dir + '/fea_{}x{}.png'.format(h, w), dpi=300)
            return x_cross_feat

    def forward(self, x:torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''

        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

def visualize_x_feat(x_feat, filename_prefix="x_feat"):
    """
    可视化 x_feat (B, H, W, C) 并保存为图片。
    - 对通道维度取平均
    - 插值到 (640, 640)
    - 保存为 filename_prefix + _{i}.png
    """
    # 确保输入是 torch.Tensor
    if not isinstance(x_feat, torch.Tensor):
        x_feat = torch.tensor(x_feat)

    # (B, H, W, C) -> (B, C, H, W)
    # x_feat = x_feat.permute(0, 3, 1, 2)

    # 通道平均 -> (B, 1, H, W)
    x_mean = x_feat.mean(dim=1, keepdim=True)

    # 插值到 (640, 640)
    x_resized = F.interpolate(x_mean, size=(640, 640), mode='bilinear', align_corners=False)

    # (B, 1, 640, 640) -> (B, 640, 640)
    x_resized = x_resized.squeeze(1)

#     # 归一化并保存
#     os.makedirs("/home/shen2/zhb/yolov5l_share_ssf/vis_results_llvip", exist_ok=True)
#     for i in range(x_resized.shape[0]):
#         img = x_resized[i].detach().cpu().numpy()
#         img = (img - img.min()) / (img.max() - img.min())*255  # 归一化到 [0, 1]

#         plt.imsave(f"/home/shen2/zhb/yolov5l_share_ssf/vis_results/{filename_prefix}_{i}.png", img, cmap='viridis')

#     print(f"✅ 可视化保存完成，共 {x_resized.shape[0]} 张图，保存在 /home/shen2/zhb/yolov5l_share_ssf/vis_results/{filename_prefix}_{i}/ 下。")
# root_path = '03921.'
# rgb_path = '/home/shen2/zhb/v5_atten_mamba/images/visible/FLIR_'+ root_path + 'jpg'
# ir_path = '/home/shen2/zhb/v5_atten_mamba/images/infrared/FLIR_'+ root_path + 'jpg'
class SSF(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            Cross:bool = True,
            scan: bool = False,
            use_learnableweights: bool = False,

            separate_return: bool = True,
            only_cm:bool = False,
            only_dbi:bool=False,

            # =================================================
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.only_cm = only_cm
        self.only_dbi = only_dbi
        if only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
            )
        if only_dbi:
            self.dbi_fusion=DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )
        if not only_dbi and not only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_rgb = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_e = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.dbi_fusion = DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )

    def forward(self,x):
        # VSSBlock=======open=====
        # temp =[]
        # temp.append(x[0].permute(0,3,1,2).contiguous())
        # temp.append(x[1].permute(0, 3,1,2).contiguous())
        # x=temp
        # ==========
        if self.only_dbi:
            x_feat=self.dbi_fusion(x)
        elif self.only_cm:
            x_feat = self.cm_fusion(x)
        elif not self.only_cm and not self.only_dbi:
            x_cross_rgb, x_share_rgb, x_cross_e, x_share_e = self.dbi_fusion(x)
            # x_feat = x_cross_rgb+x_share_rgb+x_cross_e+x_share_e
            # if x[0].shape[2]==40:
            #     visualize_x_feat(x_cross_rgb, filename_prefix=root_path + str(x[0].shape[2])+ "x_cross_rgb")
            #     visualize_x_feat(x_cross_e, filename_prefix=root_path +  str(x[0].shape[2]) + "x_cross_e")
            #     visualize_x_feat(x_share_rgb, filename_prefix=root_path +  str(x[0].shape[2])+ "x_share_rgb")
            #     visualize_x_feat(x_share_e, filename_prefix=root_path +  str(x[0].shape[2])+ "x_share_e")
            x_fuse_rgb = []
            x_fuse_e = []

            x_fuse_rgb.append(x_cross_rgb)
            x_fuse_rgb.append(x_share_rgb)

            x_fuse_e.append(x_cross_e)
            x_fuse_e.append(x_share_e)
            

            x_rgb = self.cm_fusion_rgb(x_fuse_rgb)
            x_e = self.cm_fusion_e(x_fuse_e)
            # if x[0].shape[2]==40:
            # visualize_x_feat(x_rgb, filename_prefix=root_path + str(x[0].shape[2])+ "x_fuse_rgb")
            # visualize_x_feat(x_e, filename_prefix=root_path +  str(x[0].shape[2]) + "x_fuse_e")
            # x_feat = x_share_rgb+x_share_e
            x_new=[]
            x_new.append(x_rgb)
            x_new.append(x_e)
            x_feat = self.cm_fusion(x_new)
        return x_feat


class GridregisterBlock(nn.Module):
    def __init__(self, d_model):
        super(GridregisterBlock, self).__init__()

        self.n_spoints = 1
        self.reference_points = nn.Linear(d_model, 2)
        self.sampling_offsets = nn.Linear(d_model, self.n_spoints * 2)
        self.grids_generator = nn.Linear(self.n_spoints * 2, 2)

        # self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Concat):
                for c in m.modules():
                    if isinstance(c, nn.Conv2d):
                        init.kaiming_normal_(c.weight.data, mode='fan_out')
                        if c.bias is not None:
                            init.constant_(c.bias.data, 0)
                    elif isinstance(c, nn.BatchNorm2d):
                        init.constant_(c.weight, 1)
                        init.constant_(c.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=1.0)
                init.constant_(m.bias.data, 0.)

    def get_reference_points(self, ref_fea, device):
        _, _, H_, W_ = ref_fea.shape

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                      torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1).type_as(ref_fea)

        return ref

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ir_fea_flat = ir_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # rgb_fea_flat = rgb_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # reference_points = self.reference_points(rgb_fea_flat).sigmoid().view(bs, h * w, 2)
        # sampling_offsets = self.sampling_offsets(ir_fea_flat).sigmoid().view(bs, h * w, self.n_spoints, 2)
        # offset_normalizer = torch.tensor([h, w])
        # sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :]
        # sampling_locations = reference_points[:, :, None, :] + sampling_offsets - 0.5
        # sampling_grid = self.grids_generator(sampling_locations.view(bs, h * w, self.n_spoints * 2)).sigmoid().view(bs, h * w, 2)
        # sampling_grid = 2 * sampling_grid - 1.0
        # grid = sampling_grid.view(bs, h, w, 2).clamp(-1., 1.)
        # new_ir_fea = F.grid_sample(ir_fea, grid, align_corners=True)

        # ir_fea_flat = ir_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # rgb_fea_flat = rgb_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # reference_points = self.get_reference_points(ir_fea, ir_fea.device)
        # sampling_offsets = self.sampling_offsets(rgb_fea_flat).sigmoid().view(bs, h * w, self.n_spoints, 2)
        # sampling_locations = reference_points[:, :, None, :] + sampling_offsets - 0.5
        # sampling_grid = self.grids_generator(sampling_locations.view(bs, h * w, self.n_spoints * 2)).sigmoid().view(bs, h * w, 2)
        # sampling_grid = 2 * sampling_grid - 1.0
        # grid = sampling_grid.view(bs, h, w, 2).clamp(-1., 1.)
        # new_rgb_fea = F.grid_sample(rgb_fea, grid, align_corners=True)

        # ir_fea_flat = ir_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # rgb_fea_flat = rgb_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # mp_fea_flat = torch.cat((rgb_fea_flat, ir_fea_flat), dim=2)
        # reference_points = self.get_reference_points(rgb_fea, rgb_fea.device)
        # sampling_offsets = self.sampling_offsets(mp_fea_flat).sigmoid().view(bs, h * w, self.n_spoints, 2)
        # sampling_locations = reference_points[:, :, None, :] + sampling_offsets - 0.5
        # sampling_grid = self.grids_generator(sampling_locations.view(bs, h * w, self.n_spoints * 2)).sigmoid().view(bs, h * w, 2)
        # sampling_grid = 2 * sampling_grid - 1.0
        # grid = sampling_grid.view(bs, h, w, 2).clamp(-1., 1.)
        # new_rgb_fea = F.grid_sample(rgb_fea, grid, align_corners=True)

        # rgb
        diff_fea = rgb_fea - ir_fea
        reference_points = self.get_reference_points(rgb_fea, rgb_fea.device)
        diff_fea_flat = diff_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        sampling_offsets = self.sampling_offsets(diff_fea_flat).sigmoid().view(bs, h * w, self.n_spoints, 2)
        sampling_locations = reference_points[:, :, None, :] + sampling_offsets - 0.5
        sampling_grid = self.grids_generator(sampling_locations.view(bs, h * w, self.n_spoints * 2)).sigmoid().view(bs, h * w, 2)
        sampling_grid = 2 * sampling_grid - 1.0
        grid = sampling_grid.view(bs, h, w, 2).clamp(-1., 1.)
        new_rgb_fea = F.grid_sample(rgb_fea, grid, align_corners=True)

        # ir
        # diff_fea = ir_fea - rgb_fea
        # reference_points = self.get_reference_points(ir_fea, ir_fea.device)
        # diff_fea_flat = diff_fea.contiguous().view(bs, c, -1).permute(0, 2, 1)
        # sampling_offsets = self.sampling_offsets(diff_fea_flat).sigmoid().view(bs, h * w, self.n_spoints, 2)
        # sampling_locations = reference_points[:, :, None, :] + sampling_offsets - 0.5
        # sampling_grid = self.grids_generator(sampling_locations.view(bs, h * w, self.n_spoints * 2)).sigmoid().view(bs, h * w, 2)
        # sampling_grid = 2 * sampling_grid - 1.0
        # grid = sampling_grid.view(bs, h, w, 2).clamp(-1., 1.)
        # new_ir_fea = F.grid_sample(ir_fea, grid, align_corners=True)

        # ---------------------------------------- Points Visualization ----------------------------------------- #
        # size = '_' + str(h) + 'x' + str(w) + '_'
        # save_pth = r'/home/shen/Chenyf/FLIR-ADAS-resized/save/points'
        # torch.save([reference_points * 640.0, sampling_grid * 640.0], save_pth + os.sep + 'points' + size + '.pth')

        # --------------------------------------- Feature Visualization ----------------------------------------- #
        # save_dir = '/home/shen/Chenyf/FLIR-ADAS/visual_results/features/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb = torch.floor((fea_rgb - torch.min(fea_rgb)) / (torch.max(fea_rgb) - torch.min(fea_rgb)) * 255.0)
        # fea_rgb_gird = torch.mean(new_rgb_fea, dim=1)
        # fea_rgb_gird = torch.floor((fea_rgb_gird - torch.min(fea_rgb_gird)) / (torch.max(fea_rgb_gird) - torch.min(fea_rgb_gird)) * 255.0)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir = torch.floor((fea_ir - torch.min(fea_ir)) / (torch.max(fea_ir) - torch.min(fea_ir)) * 255.0)
        # block = [fea_rgb, fea_rgb_gird, fea_ir]
        # black_name = ['fea_rgb', 'fea_new', 'fea_ir']
        #
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(1, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=600)

        # return new_fea, grid
        return new_rgb_fea, grid
        # return new_ir_fea, grid
        # return rgb_fea, rgb_fea
        # return ir_fea, ir_fea


class ACFFBlock(nn.Module):
    def __init__(self, d_model, b=1, gamma=2):
        super(ACFFBlock, self).__init__()

        self.concat = Concat(dimension=1)
        self.DCN3x3 = DCN(d_model * 2, d_model, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deform_groups=1)
        self.conv1x1 = Conv(c1=d_model, c2=1, k=1, s=1, p=0, g=1, act=True)

        k = int(abs((math.log(d_model, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.avgpool = AdaptivePool2d(1, 1, 'avg')

        self.sigmoid = nn.Sigmoid()

        # self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Concat):
                for c in m.modules():
                    if isinstance(c, nn.Conv2d):
                        init.kaiming_normal_(c.weight.data, mode='fan_out')
                        if c.bias is not None:
                            init.constant_(c.bias.data, 0)
                    elif isinstance(c, nn.BatchNorm2d):
                        init.constant_(c.weight, 1)
                        init.constant_(c.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data, gain=1.0)
                init.constant_(m.bias.data, 0.)

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # Spatial Attention
        x1 = self.DCN3x3(self.concat([rgb_fea, ir_fea]))
        y1 = x1 * self.conv1x1(x1).sigmoid()

        # Channel Attention
        x2 = self.avgpool(y1)
        x2 = self.conv1d(x2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2 = self.sigmoid(x2)
        y2 = y1 * x2.expand_as(y1)

        return y2
    


class Attention_Crossmodal(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(Attention_Crossmodal, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj_vis = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_vis = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_vis = nn.Linear(d_model, h * self.d_v)  # value projection

        self.que_proj_ir = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj_ir = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj_ir = nn.Linear(d_model, h * self.d_v)  # value projection

        self.out_proj_vis = nn.Linear(h * self.d_v, d_model)  # output projection
        self.out_proj_ir = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.init_weights()
        self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        b_s, nq = rgb_fea_flat.shape[:2]
        nk = rgb_fea_flat.shape[1]
        lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_init = lambda_init_fn(1)
        lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init

        lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(rgb_fea_flat)
        lambda_init = lambda_init_fn(1)
        lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init

        # Self-Attention
        rgb_fea_flat = self.LN1(rgb_fea_flat)
        q_vis = self.que_proj_vis(rgb_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_vis = self.key_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_vis = self.val_proj_vis(rgb_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        ir_fea_flat = self.LN2(ir_fea_flat)
        q_ir = self.que_proj_ir(ir_fea_flat).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k_ir = self.key_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v_ir = self.val_proj_ir(ir_fea_flat).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)


        att_vis = torch.matmul(q_vis, k_vis) / np.sqrt(self.d_k)
        att_ir = torch.matmul(q_ir, k_ir) / np.sqrt(self.d_k)

        # get attention matrix
        att_vis = torch.softmax(att_vis, -1)
        att_vis = self.attn_drop(att_vis)
        att_ir = torch.softmax(att_ir, -1)
        att_ir = self.attn_drop(att_ir)


        # output
        out_vis_ir = torch.matmul(att_vis, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis_vis = torch.matmul(att_vis, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_vis = out_vis_vis + lambda_rgb_full * out_vis_ir
        out_vis = self.resid_drop(self.out_proj_vis(out_vis)) # (b_s, nq, d_model)
        out_ir_vis = torch.matmul(att_ir, v_vis).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir_ir = torch.matmul(att_ir, v_ir).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                       self.h * self.d_v)  # (b_s, nq, h*d_v)
        out_ir = out_ir_vis + lambda_ir_full * out_ir_ir
        out_ir = self.resid_drop(self.out_proj_ir(out_ir)) # (b_s, nq, d_model)

        return [out_vis, out_ir]

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
class Iterative_Differential_TransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=4):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(Iterative_Differential_TransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.crossatt_ir = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.crossatt_rgb = Attention_Crossmodal(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        # self.dif_feat_ir = MultiheadDiffAttn(None, d_model, 1, h)
        # self.dif_feat_rgb = MultiheadDiffAttn(None, d_model, 1, h)

        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()
        self.lambda_q_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_rgb2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k_ir2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]
        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        temp_rgb = rgb_fea_flat
        temp_ir = ir_fea_flat
        dif_fea_rgb = x[0]
        dif_fea_ir = x[1]
        lambda_rgb_1 = torch.exp(torch.sum(self.lambda_q_rgb1 * self.lambda_k_rgb1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_rgb_2 = torch.exp(torch.sum(self.lambda_q_rgb2 * self.lambda_k_rgb2, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_1 = torch.exp(torch.sum(self.lambda_q_ir1 * self.lambda_k_ir1, dim=-1).float()).type_as(
            rgb_fea_flat)
        lambda_ir_2 = torch.exp(torch.sum(self.lambda_q_ir2 * self.lambda_k_ir2, dim=-1).float()).type_as(
            rgb_fea_flat)


        for loop in range(self.loops):

            lambda_init = lambda_init_fn(loop)
            lambda_rgb_full = lambda_rgb_1 - lambda_rgb_2 + lambda_init
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.crossatt_ir([rgb_fea_flat, ir_fea_flat])
            dif_fea_ir = ir_fea_out - lambda_rgb_full * rgb_fea_out
            # dif_fea_ir = ir_fea_out - rgb_fea_out
            rgb_fea_flat = self.coefficient5(rgb_fea_flat) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_ir)))

        for loop in range(self.loops):
            lambda_init = lambda_init_fn(loop)
            lambda_ir_full = lambda_ir_1 - lambda_ir_2 + lambda_init
            # with Learnable Coefficient
            temp_rgb_fea_out, temp_ir_fea_out = self.crossatt_rgb([temp_rgb, temp_ir])
            dif_fea_rgb = temp_rgb_fea_out - lambda_ir_full * temp_ir_fea_out
            # dif_fea_rgb = temp_rgb_fea_out - temp_ir_fea_out
            temp_ir = self.coefficient5(temp_ir) + self.coefficient6(self.mlp_vis(self.LN2(dif_fea_rgb)))


        rgb_fea_final = rgb_fea_out + dif_fea_ir
        ir_fea_final = temp_ir_fea_out + dif_fea_rgb

        # return [dif_fea_rgb, dif_fea_ir]
        return [rgb_fea_final,ir_fea_final]


class Iterative_Differential_TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super(Iterative_Differential_TransformerFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))

        # downsampling
        # self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.maxpool = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))

        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.crosstransformer = nn.Sequential(*[Iterative_Differential_TransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # Concat
        # self.concat = Concat(dimension=1)
        self.fused = NiNfusion(d_model*2, d_model)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):

        rgb_fea = x[0]
        ir_fea = x[1]

        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        #new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis

        #new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir

        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea

        new_fea = self.fused([new_rgb_fea, new_ir_fea])
        # new_fea = self.conv1x1_out(new_fea)

        # ------------------------- feature visulization -----------------------#
        # save_dir = '/home/shen/Chenyf/FLIR-align-3class/feature_save/'
        # fea_rgb = torch.mean(rgb_fea, dim=1)
        # fea_rgb_CFE = torch.mean(rgb_fea_CFE, dim=1)
        # fea_rgb_new = torch.mean(new_rgb_fea, dim=1)
        # fea_ir = torch.mean(ir_fea, dim=1)
        # fea_ir_CFE = torch.mean(ir_fea_CFE, dim=1)
        # fea_ir_new = torch.mean(new_ir_fea, dim=1)
        # fea_new = torch.mean(new_fea, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After CFE', 'fea_rgb skip', 'fea_ir', 'fea_ir After CFE', 'fea_ir skip', 'fea_ir NiNfusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        # plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)
        # -----------------------------------------------------------------------------#

        return new_fea
    
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)
        self.add = Add()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        out = self.add([rgb_fea_out, ir_fea_out])

        return out

###kanfusion

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=8):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.register_parameter(
            'fouriercoeffs',
            nn.Parameter(
                torch.randn(2, outdim, inputdim, gridsize) /
                (np.sqrt(inputdim) * np.sqrt(self.gridsize))
            )
        )

    def forward(self, x_main, x_aux):  # 接收两个模态的输入
        device = x_main.device
        dtype = x_main.dtype
        fouriercoeffs = self.fouriercoeffs.to(device=device, dtype=dtype)

        b, c, h, w = x_main.shape
        x_main = x_main.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)  # [B*H*W, C]
        x_aux = x_aux.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)

        k = torch.arange(1, self.gridsize + 1, device=device, dtype=dtype).reshape(1, 1, 1, self.gridsize)

        c = torch.cos(k * x_main.view(x_main.shape[0], 1, x_main.shape[1], 1))
        s = torch.sin(k * x_aux.view(x_aux.shape[0], 1, x_aux.shape[1], 1))

        # c = torch.sin(k * x_main.view(x_main.shape[0], 1, x_main.shape[1], 1))
        # s = torch.cos(k * x_aux.view(x_aux.shape[0], 1, x_aux.shape[1], 1))

        c = c.reshape(1, x_main.shape[0], x_main.shape[1], self.gridsize)
        s = s.reshape(1, x_aux.shape[0], x_aux.shape[1], self.gridsize)

        fourier_input = torch.cat([c, s], dim=0)  # shape: [2, B*H*W, C, G]
        y = torch.einsum("dbik,djik->bj", fourier_input, fouriercoeffs)
        y = y.view(b, h, w, self.outdim).permute(0, 3, 1, 2)
        return y


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class NaiveFourierKANLayer1(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=8):#benlai 8
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        # 初始化参数，确保类型正确
        self.register_parameter(
            'fouriercoeffs',
            nn.Parameter(
                torch.randn(2, outdim, inputdim, gridsize) /
                (np.sqrt(inputdim) * np.sqrt(self.gridsize))
            )
        )

    def to(self, device):
        # 确保继承的to方法被调用
        super(NaiveFourierKANLayer, self).to(device)
        # 明确将fouriercoeffs移到指定设备
        if self.fouriercoeffs is not None:
            self.fouriercoeffs = nn.Parameter(self.fouriercoeffs.to(device))
        return self

    def forward(self, x):
        # 确保权重在正确的设备和类型上
        device = x.device
        dtype = x.dtype
        self.fouriercoeffs = self.fouriercoeffs.to(device=device, dtype=dtype)

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, c)
        x = x.view(-1, self.inputdim)  # (b*h*w, c)

        # 生成k并确保在正确的设备和类型上
        k = torch.arange(1, self.gridsize + 1, device=device, dtype=dtype).reshape(1, 1, 1, self.gridsize)

        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))

        # 确保拼接和计算都在正确的设备和类型上
        fourier_input = torch.cat([c, s], dim=0)
        y = torch.einsum("dbik,djik->bj", fourier_input, self.fouriercoeffs)

        # 重塑输出
        y = y.view(b, h, w, self.outdim)
        y = y.permute(0, 3, 1, 2)

        return y

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=8):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.register_parameter(
            'fouriercoeffs',
            nn.Parameter(
                torch.randn(2, outdim, inputdim, gridsize) /
                (np.sqrt(inputdim) * np.sqrt(self.gridsize))
            )
        )

    def forward(self, x_main, x_aux):  # 接收两个模态的输入
        device = x_main.device
        dtype = x_main.dtype
        fouriercoeffs = self.fouriercoeffs.to(device=device, dtype=dtype)

        b, c, h, w = x_main.shape
        x_main = x_main.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)  # [B*H*W, C]
        x_aux = x_aux.permute(0, 2, 3, 1).contiguous().view(-1, self.inputdim)

        k = torch.arange(1, self.gridsize + 1, device=device, dtype=dtype).reshape(1, 1, 1, self.gridsize)

        c = torch.cos(k * x_main.view(x_main.shape[0], 1, x_main.shape[1], 1))
        s = torch.sin(k * x_aux.view(x_aux.shape[0], 1, x_aux.shape[1], 1))

        # c = torch.sin(k * x_main.view(x_main.shape[0], 1, x_main.shape[1], 1))
        # s = torch.cos(k * x_aux.view(x_aux.shape[0], 1, x_aux.shape[1], 1))

        c = c.reshape(1, x_main.shape[0], x_main.shape[1], self.gridsize)
        s = s.reshape(1, x_aux.shape[0], x_aux.shape[1], self.gridsize)

        fourier_input = torch.cat([c, s], dim=0)  # shape: [2, B*H*W, C, G]
        y = torch.einsum("dbik,djik->bj", fourier_input, fouriercoeffs)
        y = y.view(b, h, w, self.outdim).permute(0, 3, 1, 2)
        return y

import torch
import torch.nn as nn
import numpy as np





class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.shared_dim_reduction = nn.ModuleList([
            nn.Linear(input_dim // groups, (input_dim // groups) // 2)
            for _ in range(groups)
        ])
        self.base_conv = nn.ModuleList([DepthwiseSeparableConv(
            input_dim // groups,
            output_dim // groups,
            kernel_size,
            stride,
            padding
        ) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])
        self.fourier_kan = nn.ModuleList([
            NaiveFourierKANLayer(input_dim // groups, output_dim // groups)
            for _ in range(groups)
        ])
        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups) for _ in range(groups)])
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)
        # self.gsr=GaussianReLUKANLayer(input_dim,grid_size,output_dim)
        self.weight_generator = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_dim // groups, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),
                nn.Softmax(dim=1)
            ) for _ in range(groups)
        ])
        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        self.dwconv = DepthwiseSeparableConv(input_dim, output_dim, padding=1)

    def forward_fast_kan(self, rgb_x, ir_x, group_index):
        device = rgb_x.device
        rgb_norm = self.layer_norm[group_index](rgb_x)
        ir_norm = self.layer_norm[group_index](ir_x)

        rgb_base = self.base_conv[group_index](rgb_norm)
        ir_base = self.base_conv[group_index](ir_norm)

        # rgb_base=self.base_conv[group_index](self.base_activation(rgb_x))
        # ir_base=self.base_conv[group_index](self.base_activation(ir_x))
        rgb_spline = self.spline_conv[group_index](
            self.rbf(self.layer_norm[group_index](rgb_x)).moveaxis(-1, 2).flatten(1, 2)
        )
        ir_spline = self.spline_conv[group_index](
            self.rbf(self.layer_norm[group_index](ir_x)).moveaxis(-1, 2).flatten(1, 2)
        )
        # rgb_fourier = self.fourier_kan[group_index](rgb_x, ir_x)  # cos(RGB)+sin(IR)
        # ir_fourier = self.fourier_kan[group_index](ir_x, rgb_x)  # cos(IR)+sin(RGB)

        rgb_fourier = self.fourier_kan[group_index](rgb_norm, ir_norm)  # cos(RGB)+sin(IR)
        ir_fourier = self.fourier_kan[group_index](ir_norm, rgb_norm)
        # rgb_out = rgb_spline + rgb_fourier+rgb_base
        # ir_out = ir_spline + ir_fourier+ir_base
        rgb_weights = self.weight_generator[group_index](rgb_norm)  # [B, 3, 1, 1]
        ir_weights = self.weight_generator[group_index](ir_norm)

        # 融合三个分支
        rgb_out = (rgb_weights[:, 0:1] * rgb_base +
                   rgb_weights[:, 1:2] * rgb_spline +
                   rgb_weights[:, 2:3] * rgb_fourier)

        ir_out = (ir_weights[:, 0:1] * ir_base +
                  ir_weights[:, 1:2] * ir_spline +
                  ir_weights[:, 2:3] * ir_fourier)
        return rgb_out, ir_out

    def forward(self, x):
        rgb_fea, ir_fea = x[0], x[1]  # [B, L, C]
        bs, nx, c = rgb_fea.size()
        h = w = int(math.sqrt(nx))

        rgb_fea = rgb_fea.transpose(1, 2).contiguous().view(bs, c, h, w)
        ir_fea = ir_fea.transpose(1, 2).contiguous().view(bs, c, h, w)

        split_rgb = torch.split(rgb_fea, self.inputdim // self.groups, dim=1)
        split_ir = torch.split(ir_fea, self.inputdim // self.groups, dim=1)

        rgb_outputs, ir_outputs = [], []
        for i, (r, t) in enumerate(zip(split_rgb, split_ir)):
            rgb_out, ir_out = self.forward_fast_kan(r.clone(), t.clone(), i)
            rgb_outputs.append(rgb_out)
            ir_outputs.append(ir_out)

        out_rgb = self.dwconv(torch.cat(rgb_outputs, dim=1)).view(bs, self.outdim, -1).transpose(1, 2)
        out_ir = self.dwconv(torch.cat(ir_outputs, dim=1)).view(bs, self.outdim, -1).transpose(1, 2)


        return [out_rgb, out_ir]

class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0):
        kernel_size = (1, kernel_size)  # 使用1×K的卷积核
        padding = (0, padding)
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, nn.InstanceNorm2d,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout)


class KanConvBlock(nn.Module):
    def __init__(self, d_model, block_exp, attn_pdrop=.1, resid_pdrop=.1, k=3, h=8, loops_num=1, e=0.5,
                 dropoutrate=0.25):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)
        """
        super(KanConvBlock, self).__init__()
        # c_=int(d_model*e)
        self.loops = loops_num
        d_k = d_model
        d_v = d_model
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.FFKConv= FastKANConv2DLayer(d_model, d_model, kernel_size=k, padding=k // 2)
        # self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        # self.crossatt = FastKANConv2DLayer(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),

                                     # nn.SiLU(),  # changed from GELU
                                     nn.GELU(),  # changed from GELU
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop),
                                     )
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    # nn.SiLU(),  # changed from GELU
                                    nn.GELU(),  # changed from GELU
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop),
                                    )
        self.mlp = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                 # nn.SiLU(),  # changed from GELU
                                 nn.GELU(),  # changed from GELU
                                 nn.Linear(block_exp * d_model, d_model),
                                 nn.Dropout(resid_pdrop),
                                 )

        # Layer norm
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(resid_pdrop)
        # Learnable Coefficient
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()
        self.coefficient5 = LearnableCoefficient()
        self.coefficient6 = LearnableCoefficient()
        self.coefficient7 = LearnableCoefficient()
        self.coefficient8 = LearnableCoefficient()

    def forward(self, x):
        rgb_fea_flat = x[0]
        ir_fea_flat = x[1]

        assert rgb_fea_flat.shape[0] == ir_fea_flat.shape[0]
        bs, nx, c = rgb_fea_flat.size()
        h = w = int(math.sqrt(nx))

        for loop in range(self.loops):
            # with Learnable Coefficient
            rgb_fea_out, ir_fea_out = self.FFKConv([rgb_fea_flat, ir_fea_flat])  # 1  256 128
            ####ablation
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])  # 1  256 128
            rgb_att_out = self.coefficient1(rgb_fea_flat) + self.coefficient2(rgb_fea_out)
            ir_att_out = self.coefficient3(ir_fea_flat) + self.coefficient4(ir_fea_out)
            rgb_att_out = self.LN1(rgb_att_out)
            ir_att_out = self.LN2(ir_att_out)
            rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.dropout(rgb_att_out)))
            ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.dropout(ir_att_out)))

            # rgb_fea_flat = self.coefficient5(rgb_att_out) + self.coefficient6(self.mlp_vis(self.LN2(rgb_att_out)))
            # ir_fea_flat = self.coefficient7(ir_att_out) + self.coefficient8(self.mlp_ir(self.LN2(ir_att_out)))
            #######
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_att_out = self.LN1(rgb_att_out)
            # ir_att_out = self.LN2(ir_att_out)
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.dropout(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.dropout(ir_att_out))
            # without Learnable Coefficient
            # rgb_fea_out, ir_fea_out = self.crossatt([rgb_fea_flat, ir_fea_flat])
            # rgb_att_out = rgb_fea_flat + rgb_fea_out
            # ir_att_out = ir_fea_flat + ir_fea_out
            # rgb_fea_flat = rgb_att_out + self.mlp_vis(self.LN2(rgb_att_out))
            # ir_fea_flat = ir_att_out + self.mlp_ir(self.LN2(ir_att_out))

        return [rgb_fea_flat, ir_fea_flat]


class KanConvFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, block_exp=4, n_layer=1, resid_pdrop=0.1):
        super(KanConvFusionBlock, self).__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        # d_k = d_model
        # d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        ####4.8
        # self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        # self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        ###4.8
        self.avgpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(self.vert_anchors, self.horz_anchors, 'max')

        # LearnableCoefficient
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()

        # init weights
        self.apply(self._init_weights)

        # cross transformer
        self.Kanconv = nn.Sequential(*[KanConvBlock(d_model=d_model,
                                                    block_exp=block_exp,
                                                    resid_pdrop=resid_pdrop, k=3) for layer in range(n_layer)])
        # self.dynamic_snake_conv = DysnakeConvBlock(d_model, d_model,resid_pdrop=resid_pdrop,block_exp=block_exp, k=3)
        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        # self.wcmf = WCMF(channel=d_model)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):

        rgb_fea = x[0]
        ir_fea = x[1]
        # print(f"rgb_fea",rgb_fea.shape)   [1,128,20,20]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # ------------------------- cross-modal feature fusion -----------------------#
        # new_rgb_fea = (self.avgpool(rgb_fea) + self.maxpool(rgb_fea)) / 2
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_c, new_h, new_w = new_rgb_fea.shape[1], new_rgb_fea.shape[2], new_rgb_fea.shape[3]

        # new_ir_fea = (self.avgpool(ir_fea) + self.maxpool(ir_fea)) / 2
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))
        # print("Original shape:", new_rgb_fea.shape)  #[1,128,16,16]
        ###4.8
        # rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis  # blc
        # ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir  #
        ###4.8
        rgb_fea_flat = new_rgb_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1)   # blc
        ir_fea_flat = new_ir_fea.contiguous().view(bs, new_c, -1).permute(0, 2, 1)   #

        rgb_fea_flat, ir_fea_flat = self.Kanconv([rgb_fea_flat, ir_fea_flat])  # bx l c

        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='nearest')
        else:
            rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=([h, w]), mode='bilinear')
        new_rgb_fea = rgb_fea_CFE + rgb_fea
        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        if self.training == True:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='nearest')
        else:
            ir_fea_CFE = F.interpolate(ir_fea_CFE, size=([h, w]), mode='bilinear')
        new_ir_fea = ir_fea_CFE + ir_fea
        # print("After view1:", new_ir_fea.shape)
        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea











class Multi_MSDeformableNeighborhoodAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        offset_range_factor=1.0,
        stride=1,
        use_pe=True,
        dwc_pe=True,
        no_off=False,
        fixed_pe=False,
        is_causal: bool = False,
        rel_pos_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        n_head_channels = dim // num_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = num_heads
        self.nc = n_head_channels * num_heads
        self.n_groups = num_heads
        self.n_group_channels = self.nc // self.n_groups
        self.use_pe = use_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal
        
        kk = kernel_size
        pad_size = kk // 2 if kk != stride else 0

        # --- 定义生成 Offset 的网络 (分开定义 Vis 和 Tir) ---
        def build_offset_net():
            net = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels,
                          kk, stride, pad_size, groups=self.n_group_channels),
                LayerNorm2d(self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
            )
            if self.no_off:
                for m in net.parameters():
                    m.requires_grad_(False)
            return net

        self.vis_conv_offset = build_offset_net()
        self.tir_conv_offset = build_offset_net()

        # --- 定义 Projections (分开定义 Vis 和 Tir) ---
        # Vis 分支的参数
        self.vis_proj_q = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.vis_proj_k = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.vis_proj_v = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.vis_proj_out = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.vis_rpe_table = nn.Conv2d(self.nc, self.nc, 3, 1, 1, groups=self.nc)

        # Tir 分支的参数
        self.tir_proj_q = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.tir_proj_k = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.tir_proj_v = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.tir_proj_out = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.tir_rpe_table = nn.Conv2d(self.nc, self.nc, 3, 1, 1, groups=self.nc)

        # --- Relative Positional Bias (通常这是几何先验，可以共享，也可以分开) ---
        # 这里保持共享，因为相对位置偏差主要取决于 Kernel 形状，而非模态内容
        if rel_pos_bias:
            self.rpb_h1 = nn.Parameter(torch.zeros(self.n_heads//3, (2 * self.kernel_size[0] - 1), (2 * self.kernel_size[1] - 1)))
            trunc_normal_(self.rpb_h1, std=0.02, mean=0.0, a=-2.0, b=2.0)
            self.rpb_h2 = nn.Parameter(torch.zeros(self.n_heads//3, (2 * self.kernel_size[0] - 1), (2 * self.kernel_size[1] - 1)))
            trunc_normal_(self.rpb_h2, std=0.02, mean=0.0, a=-2.0, b=2.0)
            self.rpb_h3 = nn.Parameter(torch.zeros(self.n_heads//3, (2 * (self.kernel_size[0]+2)-1), (2 * (self.kernel_size[1]+2)-1)))
            trunc_normal_(self.rpb_h3, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def _cross_attn_step(self, 
                         x_query,        # 提供 Query 的特征 (例如 Vis)
                         x_kv,           # 提供 Key/Value 的特征 (例如 Tir)
                         proj_q_layer,   # 属于 Query 模态的 Q 投影层
                         conv_off_layer, # 属于 Query 模态的 Offset 网络
                         rpe_layer,      # 属于 Query 模态的 Pos 编码
                         proj_k_layer,   # 属于 KV 模态的 K 投影层
                         proj_v_layer,   # 属于 KV 模态的 V 投影层
                         proj_out_layer  # 属于 Query 模态的 输出投影层 (因为目的是更新 Query)
                         ):
        
        B, C, H, W = x_query.size()
        dtype, device = x_query.dtype, x_query.device

        # 1. 使用 Query 模态特定的层生成 Q 和 Offset
        q = proj_q_layer(x_query)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        
        offset = conv_off_layer(q_off).contiguous()
        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)
        if pos.dtype != x_kv.dtype:
            pos = pos.to(x_kv.dtype)
        # 2. 对 KV 模态进行采样 (Input是 x_kv)
        # 这里的含义：去 x_kv (例如 Tir) 中，在 x_query (例如 Vis) 感兴趣的位置采样
        if self.no_off:
            x_sampled = F.avg_pool2d(x_kv, kernel_size=self.stride, stride=self.stride)
        else:
            x_sampled = F.grid_sample(
                input=x_kv.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],
                mode='bilinear', align_corners=True)
        
        x_sampled = x_sampled.reshape(B, C, H, W)

        # 3. 使用 KV 模态特定的层生成 K 和 V
        # 注意：这里使用的是传入的 proj_k_layer 和 proj_v_layer，它们属于 x_kv 的模态
        k = einops.rearrange(proj_k_layer(x_sampled), 'b (g c) h w -> b g h w c', g=self.n_groups)
        v = einops.rearrange(proj_v_layer(x_sampled), 'b (g c) h w -> b g h w c', g=self.n_groups)

        # 4. Q 整理格式
        q = einops.rearrange(q, 'b (g c) h w -> b g h w c', g=self.n_groups)
        q = q * self.scale

        # 5. RPE (通常跟随 Query 的空间结构)
        residual_lepe = rpe_layer(x_query) # 注意这里通常是对 Q 之前的 feat 或者 Q 之后做，原代码是对 Q 算出来的 conv 结果做

        # 原代码逻辑里 residual_lepe = self.rpe_table(q_input)，这里为了匹配原逻辑：
        # 如果 rpe_table 输入是 proj_q 后的结果：
        residual_lepe = rpe_layer(proj_q_layer(x_query)) 

        # 6. Attention 计算 (逻辑不变)
        q1, q2, q3 = q.chunk(chunks=3, dim=1)
        k1, k2, k3 = k.chunk(chunks=3, dim=1)
        v1, v2, v3 = v.chunk(chunks=3, dim=1)

        attn1 = na2d_qk(q1.contiguous(), k1.contiguous(), kernel_size=self.kernel_size, dilation=self.dilation, is_causal=self.is_causal, rpb=self.rpb_h1)
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out_h1 = na2d_av(attn1, v1.contiguous(), kernel_size=self.kernel_size, dilation=self.dilation, is_causal=self.is_causal)

        attn2 = na2d_qk(q2.contiguous(), k2.contiguous(), kernel_size=self.kernel_size, dilation=2, is_causal=self.is_causal, rpb=self.rpb_h2)
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out_h2 = na2d_av(attn2, v2.contiguous(), kernel_size=self.kernel_size, dilation=2, is_causal=self.is_causal)

        attn3 = na2d_qk(q3.contiguous(), k3.contiguous(), kernel_size=(self.kernel_size[0]+2, self.kernel_size[1]+2), dilation=self.dilation, is_causal=self.is_causal, rpb=self.rpb_h3)
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)
        out_h3 = na2d_av(attn3, v3.contiguous(), kernel_size=(self.kernel_size[0]+2, self.kernel_size[1]+2), dilation=self.dilation, is_causal=self.is_causal)

        out = einops.rearrange(torch.cat([out_h1, out_h2, out_h3], dim=1), 'b g h w c -> b (g c) h w')

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        if out.dtype != proj_out_layer.weight.dtype:
            out = out.to(proj_out_layer.weight.dtype)
        # 7. Output Projection (属于 Query 模态，因为我们要恢复到 Query 的特征空间)
        y = self.proj_drop(proj_out_layer(out))
        return y

    def forward(self, vis, tir):
        """
        [y_vis, y_tir]
        """
        # --- 分支 1: 更新 Vis (Vis Query, Tir KV) ---
        # Q: Vis Proj
        # K, V: Tir Proj
        # Out: Vis Proj
        y_vis = self._cross_attn_step(
            x_query=vis, 
            x_kv=tir, 
            proj_q_layer=self.vis_proj_q,
            conv_off_layer=self.vis_conv_offset,
            rpe_layer=self.vis_rpe_table,
            proj_k_layer=self.tir_proj_k, # 注意：Key 来自 Tir，使用 Tir 的投影
            proj_v_layer=self.tir_proj_v, # 注意：Value 来自 Tir，使用 Tir 的投影
            proj_out_layer=self.vis_proj_out
        )

        # --- 分支 2: 更新 Tir (Tir Query, Vis KV) ---
        # Q: Tir Proj
        # K, V: Vis Proj
        # Out: Tir Proj
        y_tir = self._cross_attn_step(
            x_query=tir, 
            x_kv=vis, 
            proj_q_layer=self.tir_proj_q,
            conv_off_layer=self.tir_conv_offset,
            rpe_layer=self.tir_rpe_table,
            proj_k_layer=self.vis_proj_k, # 注意：Key 来自 Vis，使用 Vis 的投影
            proj_v_layer=self.vis_proj_v, # 注意：Value 来自 Vis，使用 Vis 的投影
            proj_out_layer=self.tir_proj_out
        )

        return [y_vis, y_tir]


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        self.norm_type = LayerNorm_type
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            # self.body = WithBias_LayerNorm(dim)
            self.body = LayerNorm2d(dim, eps=1e-5)
            # eps default 1e-6 consistent to 1e-5 in Restormer

    def forward(self, x):

        if self.norm_type == 'BiasFree':
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)

        else:
            return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x






class MultiTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,kernel_size=3, 
                 ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias', rel_pos_bias = True,
                 dilation=1):
        super(MultiTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Multi_MSDeformableNeighborhoodAttention(
            dim, num_heads, kernel_size=kernel_size, dilation=dilation, rel_pos_bias=rel_pos_bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn_vis = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_tir = FeedForward(dim, ffn_expansion_factor, bias)
        self.ninfuse = NiNfusion(dim*2, dim)

    def forward(self, x):
        vis = x[0]
        tir = x[1]
        vis, tir = self.attn(self.norm1(vis),self.norm1(tir))
        vis  = x[0] + vis
        tir  = x[1] + tir
        vis = vis + self.ffn_vis(self.norm2(vis))
        tir = tir + self.ffn_tir(self.norm2(tir))
        out  = self.ninfuse([vis, tir])

        return out 