import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, z_dim, hidden_dims_dec):
        super().__init__()
        # setup the linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dims_dec[0])
        self.fc2 = nn.Linear(hidden_dims_dec[0], hidden_dims_dec[1])
        self.fc3 = nn.Linear(hidden_dims_dec[1], hidden_dims_dec[2])
        self.fc12 = nn.Linear(hidden_dims_dec[2], input_size)
        self.fc11 = nn.Linear(hidden_dims_dec[2], input_size)

        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden1 = self.relu(self.fc1(z))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.relu(self.fc3(hidden2))
        # return the parameter for the output
        loc_x = self.fc12(hidden3)
        var_x = torch.exp(self.relu(self.fc11(hidden3)))
        return loc_x, var_x