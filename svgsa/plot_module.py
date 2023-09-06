import os
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context

def plot_loss(loss, save_directory=None):

    path = os.getcwd()

    if save_directory is not None:
        path = path + save_directory

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(5, 2))
    plt.plot(loss)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.savefig(os.path.join(path,'ElboLoss.png'), bbox_inches="tight")


def umap_embedding(adata, rep, expression_list, color_label = "bulk_labels", ncols=4, vmin=-2, vmax=2, palette=None, save_directory=None):   #sc_palette

    path = os.getcwd()

    if save_directory is not None:
        path = path + save_directory

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    sc.pp.neighbors(adata, use_rep = rep)
    sc.tl.umap(adata, n_components=2)
    with rc_context({'figure.figsize': (3, 3)}):
        if palette:
            sc.pl.umap(adata, color=expression_list + [color_label], s=50, frameon=False, ncols=ncols, vmin=vmin, vmax=vmax, palette=palette, show=False)
        else:
            sc.pl.umap(adata, color=expression_list + [color_label], s=50, frameon=False, ncols=ncols, vmin=vmin, vmax=vmax, show=False)
        plt.savefig(os.path.join(path,rep+'.png'), bbox_inches="tight")