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
        chan = self.msflow.extractor.out_channels
        self.out = nn.Sequential(
            nn.Linear(chan, chan // 2),
            nn.ReLU(inplace=True),
            nn.Linear(chan // 2, chan // 4),
            nn.ReLU(inplace=True),
            nn.Linear(chan // 4, 2)
        )

        for param in self.msflow.parameters():
            param.requires_grad = False
        self.msflow.eval()

    def forward(self, img):
        with torch.no_grad():
            zs, hs = self.msflow.z_forward(img)
        x = self.fusion(zs, hs)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x

    # @torch.no_grad()
    # def predict(self, img):
    #     logits = self.forward(img)
    #     logits = torch.softmax(logits, dim=-1)
    #     preds = torch.argmax(logits, dim=-1)
    #     return preds

    @torch.no_grad()
    def predict(self, img, top_k, threshold):
        zs, hs = self.msflow.z_forward(img)
        x = self.fusion(zs, hs)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.out(x)
        logits = torch.softmax(logits, dim=-1)
        pred1 = torch.argmax(logits, dim=-1).cpu().numpy()

        zs, fuse_jac = self.msflow.fusion_flow(zs)
        outputs_list = [list() for _ in self.msflow.parallel_flows]
        size_list = []
        for lvl, z in enumerate(zs):
            size_list.append(list(z.shape[-2:]))
            logp = - 0.5 * torch.mean(z ** 2, 1)
            outputs_list[lvl].append(logp)
        score, _, _ = self.msflow.postprocess(size_list, outputs_list, img.shape[-2:], top_k=top_k)

        pred = 1 if score > threshold or pred1 == 1 else 0
        return pred


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

    def forward(self, zs, hs):
        x = self.layer1(zs[0] + hs[0])
        x = self.layer2(x + zs[1] + hs[1])
        x = self.layer3(x + zs[2] + hs[2])
        return x

