import numpy as np
import pandas as pd
import torch
import scanpy as sc
import platform
from collections import OrderedDict


def load_pbmc68k_reduced(data_name=False):
    adata =  sc.datasets.pbmc68k_reduced()
    if data_name:
        name = "pbmc68k_reduced"
        return adata, name
    return adata

def load_dataset(directory, data_name=True):

    os_name = platform.system()
    if os_name == 'Windows':
        s = '\\'
    elif os_name == 'Linux':
        s = '/'
    else:
        print('Operating system not supported')

    adata = sc.read(directory)
    if data_name:
        name = directory.split(s)[-1].split('.')[0]
        return adata, name
    return adata

def sim_mask(ngenes,ngs):
    mask_gs = torch.zeros(ngenes,ngs).cpu()
    step = np.int_(ngenes/ngs)
    for i in range(ngs):
        mask_gs[i*step:i*step+step,i] = 1
    return mask_gs

def sim_data(ncells,ngenes,ngs,mask_gs,expr,seed):
    
    torch.manual_seed(seed)
    data = torch._standard_gamma(torch.ones(ncells,ngenes)*1.5).cpu()
    mask = mask_gs*expr
    mask[mask_gs.bool()] = torch.normal(mask[mask_gs.bool()], std=0.001)
    weight = mask.clone()
    mask[mask_gs.bool()] = mask[mask_gs.bool()] - 1.0
    step_c = np.int_(ncells/mask.shape[1])
    step_g = np.int_(ngenes/mask.shape[1])
    n_digits_cell = len(str(ncells))
    n_digits_gene = len(str(ngenes))
    n_digits_gs = len(str(ngs))

    ct = np.empty(ncells, dtype='object')
    source = np.empty(ngenes, dtype='object')
    for i in range(ngs):
        if expr[i]:
            data[i*step_c:(i+1)*step_c,:] += data[i*step_c:(i+1)*step_c,:] * mask[:,i] 
        ct[i*step_c:(i+1)*step_c] = f'Gene_Set{str(i+1).zfill(n_digits_gs)}'
        source[i*step_g:(i+1)*step_g] = f'Gene_Set{str(i+1).zfill(n_digits_gs)}'

    adata = sc.AnnData(X = data.numpy())
    adata.raw = adata

    adata.obs_names = [f"Cell{str(number).zfill(n_digits_cell)}" for number in range(adata.n_obs)]
    adata.var_names = [f"Gene{str(number).zfill(n_digits_gene)}" for number in range(adata.n_vars)] 
    adata.raw.var.index = adata.var_names
    adata.obs["gs_labels"] = pd.Categorical(ct)

    target = adata.var_names

    pf = {'source': source,'target': target, 'weight': (weight).sum(axis=1).tolist()}
    gsts = pd.DataFrame(pf)

    return adata, gsts

def load_sim_data(ncells,ngenes,ngs,expr,seed=42,data_name=False):
    '''it returns
        adata: AnnData object, that is a simulated dataset.
        gsts: dictionary of gene sets activations.
        name: string, name of the dataset.
    '''
    adata, gsts = sim_data(ncells,ngenes,ngs,sim_mask(ngenes,ngs),expr,seed)
    if data_name:
        name = f"Simulated_nc{ncells}_ng{ngenes}_ngs{ngs}"
        return adata, gsts, name
    return adata, gsts

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of pathway:genes.
    min_g and max_g are optional gene set size filters.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway