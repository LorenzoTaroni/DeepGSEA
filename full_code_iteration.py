import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import pyro
import torch
import pandas as pd
from tqdm import trange

from matplotlib.pyplot import rc_context

import decoupler as dc

import svgsa

import platform

from itertools import product

plt.style.use('ggplot')

z_dim_gs = [0, 2, 5, 10, 50, 100, 200]
z_dim_uns = [0, 2, 5, 10, 50, 100, 200]
epochs = [100, 200, 500, 1000, 2000]
batch_size = [100, 300, 700]

all_permutations = list(product(z_dim_gs, z_dim_uns, epochs, batch_size))


for perm in all_permutations:

    z_dim_gs, z_dim_uns, epochs, batch_size = perm

    if z_dim_gs == 0:
        continue

    adata, data_name = svgsa.load_pbmc68k_reduced(data_name=True)

    #adata.obsm["raw"] = adata.raw.X.toarray().astype(np.float32)

    pgy = dc.get_progeny(organism = "human", top = 500)
    
    gsts_dict = pgy.groupby("source")["target"].apply(lambda x: sorted(x)).to_dict()

    pyro.enable_validation()
    #torch.autograd.set_detect_anomaly(True)

    adata_new, model, loss, lrs, N_GS_list = svgsa.fit_SVGSA(adata, gsts_dict, z_dim_gs=z_dim_gs, z_dim_uns=z_dim_uns,
                                   
                                   num_iafs =0, batch_size=batch_size, epochs=epochs, 
                                   
                                   N_GS=14, normalize = False, CUDA = True, iaf_dim=50, fixed = False)
    

    os_name = platform.system()
    if os_name == 'Windows':
        s = '\\'
    elif os_name == 'Linux':
        s = '/'
    else:
        print('Operating system not supported')

    d = f'{s}plots{s}{data_name}{s}z_dim_gs{str(z_dim_gs)}_z_dim_uns{str(z_dim_uns)}{s}epochs_{str(epochs)}{s}batch_size_{batch_size}{s}'

    svgsa.plot_loss(loss, save_directory=d)

    svgsa.plot_learning_rate_decay(lrs, save_directory=d)

    if z_dim_gs > 0:
        rep = ['X', 'X_svgsa_gs',]
    if z_dim_uns > 0:
        rep = ['X', 'X_svgsa_uns']
    if z_dim_gs > 0 and z_dim_uns > 0:
        rep = ['X', 'X_svgsa', 'X_svgsa_gs', 'X_svgsa_uns']

    svgsa.full_plot(adata_new, rep, list(N_GS_list), color_label = "bulk_labels", save_directory=d)