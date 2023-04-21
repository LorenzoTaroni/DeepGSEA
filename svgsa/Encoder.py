import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dims_enc):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dims_enc[0])
        self.bc1 = nn.BatchNorm1d(hidden_dims_enc[0])
        
        self.fc2 = nn.Linear(hidden_dims_enc[0], hidden_dims_enc[1])
        self.bc2 = nn.BatchNorm1d( hidden_dims_enc[1])
        
        self.fc3 = nn.Linear(hidden_dims_enc[1], hidden_dims_enc[2])
        self.bc3 = nn.BatchNorm1d(hidden_dims_enc[2])

        self.fc21 = nn.Linear(hidden_dims_enc[2], z_dim)
        self.fc22 = nn.Linear(hidden_dims_enc[2], z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward computation on the sample x
        hidden1 = self.relu(self.bc1(self.fc1(x)))
        hidden2 = self.relu(self.bc2(self.fc2(hidden1)))
        hidden3 = self.relu(self.bc3(self.fc3(hidden2)))
        
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden3)
        z_scale = torch.exp(self.fc22(hidden3))
        return z_loc, z_scale