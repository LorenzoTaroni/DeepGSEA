# Import necessary libraries
import torch
import pyro
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from svgsa.pathway_dispersion import select_highest_variance_gs
from svgsa.sc_dataset import SingleCellDataset
from tqdm import trange

import numpy as np

from svgsa.VAE import VAE

def fit_SVGSA(adata, gene_dict, lr=5*1e-3, seed=3, CUDA=False, epochs=10, z_dim_gs=10, z_dim_uns=15, encoder_dims_gs=[16, 8, 4], encoder_dims_uns=[256,128,64], decoder_dims=None, N_GS=20, beta=10000, fixed=False, batch_size=16, compile_JIT=False, normalize=True, num_iafs=0, iaf_dim=40, gs_list=None):

    # Set decoder dimensions if not provided
    if decoder_dims is None:
        if z_dim_uns > 0:
            decoder_dims = encoder_dims_uns[::-1]
        else:
            decoder_dims = encoder_dims_gs[::-1]

    # Set random seed for reproducibility
    pyro.set_rng_seed(seed)
    
    # Clear parameters in Pyro
    pyro.clear_param_store() 

    # Set default tensor type and generator based on CUDA availability
    if CUDA:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        generator = torch.Generator(device='cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        generator = torch.Generator()

    # Select loss function based on compile_JIT flag
    if compile_JIT:
        loss = pyro.infer.JitTraceGraph_ELBO
    else:
        loss = pyro.infer.TraceGraph_ELBO

    vr = []
    
    # Select highest variance gene sets if z_dim_gs > 0 and not fixed
    if z_dim_gs > 0 and not fixed:
        vr = list(select_highest_variance_gs(adata.X, gene_dict, N_GS, adata.var_names, normalize=normalize))
        idxs = [np.intersect1d(np.array(adata.var_names), np.array(gene_dict[k]), return_indices=True)[1] for k in vr]
        mask = torch.zeros([adata.shape[1], N_GS])
        for i in range(len(idxs)):
            mask[idxs[i], i] = 1
        print(vr)

    # If fixed and gs_list is None, select the first N_GS gene sets
    elif fixed and gs_list is None:
        vr = list(gene_dict.keys())[:N_GS]
        idxs = [np.intersect1d(np.array(adata.var_names), np.array(gene_dict[k]), return_indices=True)[1] for k in vr]
        mask = torch.zeros([adata.shape[1], N_GS])
        for i in range(len(idxs)):
            mask[idxs[i], i] = 1
        print(vr)

    # If fixed and gs_list is provided, ensure all elements are in gene_dict and create the mask
    elif fixed and gs_list is not None:
        vr = list(gene_dict.keys())
        assert all(element in vr for element in gs_list), "The list provided contains gene sets not present in the given gene set dictionary"
        vr = gs_list
        N_GS = len(vr)
        idxs = [np.intersect1d(np.array(adata.var_names), np.array(gene_dict[k]), return_indices=True)[1] for k in vr]
        mask = torch.zeros([adata.shape[1], N_GS])
        for i in range(len(idxs)):
            mask[idxs[i], i] = 1
        print(vr)

    # If no other conditions are met, create a mask of ones
    else:
        mask = torch.ones([adata.shape[1], N_GS])

    # Model initialization
    vae = VAE(mask=mask.t(), input_dim=adata.shape[1],
              z_dim_gs=z_dim_gs, z_dim_uns=z_dim_uns,
              hidden_dims_dec=decoder_dims, 
              hidden_dims_enc_gs=encoder_dims_gs, 
              hidden_dims_enc_uns=encoder_dims_uns,
              beta=beta, use_cuda=CUDA, n_gs=N_GS,
              batch_size=batch_size, num_iafs=num_iafs, iaf_dim=iaf_dim)

    model = vae.model
    guide = vae.guide

    # Input initialization
    input = SingleCellDataset(adata, N_GS, normalize=normalize)
    input_dataloader = DataLoader(input, batch_size=batch_size, shuffle=True, generator=generator)

    # Initialize SVI (Stochastic Variational Inference)
    svi = pyro.infer.SVI(model=model,
                         guide=guide,
                         optim=pyro.optim.ClippedAdam({"lr": lr, "lrd": np.exp(np.log10(1e-3 / lr) / epochs)}),
                         loss=loss())

    # Initialize progress bar
    t = trange(epochs, desc='Bar desc', leave=True)

    losses = []
    lrs = []
    
    # Training loop
    for epoch in t:
        epoch_loss = 0.
        # Iterate over each mini-batch
        for x in input_dataloader:
            if CUDA:
                x = x.cuda()
            # ELBO gradient step and accumulate loss
            epoch_loss += svi.step(x.transpose(1,0)) / batch_size
            # Keeping track of learning rate
            current_lr = svi.optim.get_state()['encoder_gs$$$bc1.weight']['param_groups'][0]['lr']

        losses.append(epoch_loss)
        lrs.append(current_lr)
        t.set_description("Epoch loss %f" % epoch_loss)
        t.refresh()
        
    # Process and save results to adata based on CUDA availability
    if CUDA:
        adata.obsm["X_svgsa"] = vae.sample_hidden_dims(input.counts, sample=False).cpu().detach().numpy()
        adata.obsm["X_reconstructed"] = vae.reconstruct_cell(input.counts).cpu().detach().numpy()

        if z_dim_gs > 0:
            adata.obsm["last_node"] = vae.return_gs(input.counts, return_geneset=True).cpu().detach().numpy()

        if not normalize:
            input.raw_counts = input.counts

        for i in range(len(vr)):
            adata.obs[vr[i]] = scale((input.raw_counts[i,:,:] * mask[:,i]).sum(axis=1).cpu().detach())

    else:
        adata.obsm["X_svgsa"] = vae.sample_hidden_dims(input.counts, sample=False).detach().numpy()
        adata.obsm["X_reconstructed"] = vae.reconstruct_cell(input.counts).detach().numpy()

        if z_dim_gs > 0:
            adata.obsm["last_node"] = vae.return_gs(input.counts, return_geneset=True).detach().numpy()

        if not normalize:
            input.raw_counts = input.counts

        for i in range(len(vr)):
            adata.obs[vr[i]] = scale((input.raw_counts[i,:,:] * mask[:,i]).sum(axis=1).detach())
        
    # Split and save latent space representation into gene set and unsupervised parts
    adata.obsm["X_svgsa_gs"] = adata.obsm["X_svgsa"][:, 0:z_dim_gs]
    adata.obsm["X_svgsa_uns"] = adata.obsm["X_svgsa"][:, z_dim_gs:(z_dim_gs + z_dim_uns + 1)]
    
    return adata, svi, losses, lrs, vr
