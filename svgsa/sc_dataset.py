import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import scale


class SingleCellDataset(Dataset):
    def __init__(self, adata, n_gs, normalize = True):
        if normalize:
            self.data = np.clip(scale(adata.X / adata.X.sum(axis=1).reshape([-1,1]) * 10**6).astype(np.float32), -10,10)
            self.raw = adata.X
            self.raw_counts = torch.tensor(self.raw).unsqueeze(0).repeat(n_gs, 1, 1)
        else:
            self.data = np.clip(adata.X.astype(np.float32), -10,10)
            self.raw = adata.raw
        self.counts = torch.tensor(self.data).unsqueeze(0).repeat(n_gs, 1, 1)
        self.genes = adata.var_names
        self.cells = adata.obs_names
        self.n_genes = adata.n_vars
        self.n_cells = adata.n_obs

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return self.counts[:,idx,:]