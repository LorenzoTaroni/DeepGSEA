# Code adapted from https://github.com/pytorch/pytorch/issues/36591

import torch
import torch.nn as nn
import math


class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size,mask = None, batch_size = 128):
        super(LinearWithChannel, self).__init__()

        self._mask = mask
        
        if self._mask is not None:
            self._mask.requires_grad = False
            self._mask = self._mask.reshape([channel_size,input_size,1])
        # initialize weights
        self.weight = torch.nn.Parameter(torch.randn(channel_size, input_size, output_size))
        self.bias = torch.nn.Parameter(torch.randn(channel_size, 1, output_size))

        # change weights to kaiming
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.xavier_uniform_(weights)
        torch.nn.init.uniform_(bias)

    def forward(self, x):
        if self._mask is not None:
            self.weight.data = self.weight.data * self._mask
            #self.weight.mul_(self._mask)
            
        return (torch.bmm(x, self.weight) + self.bias).transpose(1,0)