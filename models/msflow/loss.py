import torch
import torch.nn as nn


class MSFlowLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_list, jac):
        loss = 0.

        for z in z_list:
            loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
        loss = loss - jac
        loss = loss.mean()
        return loss
