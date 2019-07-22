
from __future__ import absolute_import
from __future__ import division

from torch import nn
import torch
import pdb


class net_classification(nn.Module):
    def __init__(self):
        super(net_classification, self).__init__()
        self.sig = nn.Sigmoid()

        self.Conv2D = nn.Conv2d(2048, 100, (1, 1))
        self.linear200 = nn.Linear(200, 100)
        self.linear100 = nn.Linear(100, 25)
        self.linear25 = nn.Linear(25, 1)
        self.drop = nn.Dropout(p=0.3)
        self.Relu = nn.ReLU()

    def forward(self, input1, input2):
        merged_add = input1 + input2
        merged_sub = input1 - input2

        merged_add = self.Conv2D(merged_add)
        merged_sub = self.Conv2D(merged_sub)

        merged = torch.cat((merged_add, merged_sub), dim=-1)
        merged = torch.flatten(merged, start_dim=1)

        merged = self.linear200(merged)
        merged = self.Relu(merged)
        merged = self.drop(merged)
        merged = self.linear100(merged)
        merged = self.Relu(merged)
        merged = self.drop(merged)
        merged = self.linear25(merged)
        out = self.sig(merged)

        return out
