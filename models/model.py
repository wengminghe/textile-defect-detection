import torch
import torch.nn as nn

from .backbones import BACKBONES
from models.msflow.msflow import MSFlow


class Model(nn.Module):
    def __init__(self, backbone, c_conds, parallel_blocks, pool_type='avg', clamp_alpha=1.9, **kwargs):
        super(Model, self).__init__()
        self.msflow = MSFlow(backbone, c_conds, parallel_blocks, pool_type, clamp_alpha, **kwargs)
        self.fusion = Fusion(self.msflow.extractor.out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

        for param in self.msflow.parameters():
            param.requires_grad = False
        self.msflow.eval()

    def forward(self, img):
        with torch.no_grad():
            zs, hs = self.msflow.z_forward(img)
        x = self.fusion.layer2(zs[0] + hs[0])
        x = self.fusion.layer3(x + zs[1] + hs[1])
        x = self.fusion.layer4(x + zs[2] + hs[2])
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

    @torch.no_grad()
    def predict(self, img):
        logits = self.forward(img)
        logits = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return preds


class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2] * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2] * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, hs):
        x = self.layer1(hs[0])
        x = self.layer2(x + hs[1])
        x = self.layer3(x + hs[2])
        return x


class Model0(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(Model0, self).__init__()
        self.backbone = BACKBONES[backbone](pretrained=True)
        self.fusion = Fusion(self.backbone.out_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        chan = self.backbone.out_channels[-1] * 2
        self.out = nn.Sequential(
            nn.Linear(chan, chan // 2),
            nn.ReLU(inplace=True),
            nn.Linear(chan // 2, chan // 4),
            nn.ReLU(inplace=True),
            nn.Linear(chan // 4, 2)
        )

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, img):
        with torch.no_grad():
            hs = self.backbone(img)
        x = self.fusion(hs)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

    @torch.no_grad()
    def predict(self, img):
        logits = self.forward(img)
        logits = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return preds
