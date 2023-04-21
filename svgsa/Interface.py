import torch
import pyro
from torch.utils.data import DataLoader
from svgsa.pathway_dispersion import select_highest_variance_gs
from svgsa.sc_dataset import SingleCellDataset
from tqdm import trange

import numpy as np

from svgsa.VAE import VAE

def fit_SVGSA(adata, gene_dict, lr = 0.001, seed = 3, CUDA = False, epochs = 10, z_dim_gs = 10, z_dim_uns = 15, decoder_dims = [32,64,128], encoder_dims_uns = [128,64,32], encoder_dims_gs = [8,4,4], N_GS = 20, fixed = False, batch_size = 16, compile_JIT = False, normalize = True, num_iafs = 0, iaf_dim = 40):

    pyro.set_rng_seed(seed)
    
    pyro.clear_param_store() 

    if CUDA:
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        generator = torch.Generator(device='cuda')
    else:
        generator = torch.Generator()

    if compile_JIT:
        loss = pyro.infer.JitTraceGraph_ELBO
    else:
        loss = pyro.infer.TraceGraph_ELBO
    
    if z_dim_gs > 0 and not fixed:

        vr = select_highest_variance_gs(adata.X, gene_dict, N_GS, adata.var_names,normalize= normalize)
        idxs = [np.intersect1d(np.array(adata.var_names), np.array(gene_dict[k]), return_indices=True)[1] for k in vr]
        mask = torch.zeros([adata.shape[1], N_GS])
        for i in range(len(idxs)):
            mask[idxs[i], i] = 1

    elif fixed:

        idxs = [np.intersect1d(np.array(adata.var_names), np.array(gene_dict[k]), return_indices=True)[1] for k in gene_dict.keys()]
        mask = torch.zeros([adata.shape[1], N_GS])
        for i in range(len(idxs)):
            mask[idxs[i], i] = 1
        
    else:

        mask = mask = torch.ones([adata.shape[1], N_GS])


    print(mask.shape)

    # Model initialization
    
    vae = VAE(mask = mask.t(), input_dim = mask.shape[0],
              z_dim_gs = z_dim_gs, z_dim_uns = z_dim_uns,
              hidden_dims_dec = decoder_dims, 
              hidden_dims_enc_gs = encoder_dims_gs, 
              hidden_dims_enc_uns = encoder_dims_uns,
              use_cuda = CUDA, n_gs = N_GS, 
              batch_size = batch_size, num_iafs = num_iafs,iaf_dim = iaf_dim )

    model = vae.model
    guide = vae.guide


    # Input initialization

    input = SingleCellDataset(adata, N_GS, normalize = normalize)

    print(input.shape)

    input_dataloader = DataLoader(input, batch_size=batch_size, shuffle=True, generator=generator)




    svi = pyro.infer.SVI(model=model,
                     guide=guide,
                     optim=pyro.optim.ClippedAdam({"lr": lr}),
                     loss= loss())
    t = trange(epochs, desc='Bar desc', leave=True)
    
    
    
    losses = []
    
    for epoch in t:
        
        
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x in input_dataloader:
            if CUDA:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x.transpose(1,0))
        losses.append(epoch_loss / batch_size)
        t.set_description("Epoch loss %f" % epoch_loss)
        t.refresh()
        
    if CUDA:
        adata.obsm["X_svgsa"] = vae.sample_hidden_dims(input.counts, sample = False).cpu().detach().numpy()
        adata.obsm["X_reconstructed"] = vae.reconstruct_cell(input.counts).cpu().detach().numpy()
    else:
        adata.obsm["X_svgsa"] = vae.sample_hidden_dims(input.counts, sample = False).detach().numpy()
        adata.obsm["X_reconstructed"] = vae.reconstruct_cell(input.counts).detach().numpy()
        
    adata.obsm["X_svgsa_gs"] = adata.obsm["X_svgsa"][:, 0:z_dim_gs]
    adata.obsm["X_svgsa_uns"] = adata.obsm["X_svgsa"][:, z_dim_gs:(z_dim_gs + z_dim_uns + 1)]
    
    return adata, svi, losses