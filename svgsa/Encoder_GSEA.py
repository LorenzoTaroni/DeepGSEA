import torch
import torch.nn as nn
from svgsa.LinearWithChannel import LinearWithChannel

class Encoder_GSEA(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims_enc, channels, mask, batch_size):
        super().__init__()

        self._channels = channels
        
        # setup the three linear transformations used
        self.fc1 = LinearWithChannel(input_dim, hidden_dims_enc[0], channels, mask.unsqueeze(-1),batch_size)
        self.bc1 = nn.BatchNorm1d(channels)
        
        self.fc2 = LinearWithChannel(hidden_dims_enc[0], hidden_dims_enc[1], channels, None, batch_size)
        self.bc2 = nn.BatchNorm1d(channels)
        
        self.fc3 = LinearWithChannel(hidden_dims_enc[1], hidden_dims_enc[2], channels, None, batch_size)
        self.bc3 = nn.BatchNorm1d(channels)
        
        self.fc4 = LinearWithChannel(hidden_dims_enc[2], 1, channels, None, batch_size)
        self.bc4 = nn.BatchNorm1d(channels)

        self.fc21 = nn.Linear(channels, z_dim)
        self.fc22 = nn.Linear(channels, z_dim)

        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, x, return_geneset = False):
        # define the forward computation on the sample x
        hidden1 = self.relu(self.bc1(self.fc1(x))).transpose(1,0)
        hidden2 = self.relu(self.bc2(self.fc2(hidden1))).transpose(1,0)
        hidden3 = self.relu(self.bc3(self.fc3(hidden2))).transpose(1,0)
        # Gene set enrichment score layer
        hidden4 = torch.tanh(self.bc4(self.fc4(hidden3))).squeeze()
        # return scores to visualize gene set predicted expressions
        if return_geneset:
            return hidden4

        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden4)
        z_scale = torch.exp(self.fc22(hidden4))
        
        return z_loc, z_scale