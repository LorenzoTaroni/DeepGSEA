import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from matplotlib.lines import Line2D
import seaborn as sns

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


def plot_learning_rate_decay(lrs, save_directory=None):

    path = os.getcwd()

    if save_directory is not None:
        path = path + save_directory

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(5, 2))
    plt.plot(lrs)
    plt.xlabel("SVI step")
    plt.ylabel("lr decay")
    plt.savefig(os.path.join(path,'lr_decay.png'), bbox_inches="tight")


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




def full_plot(adata, rep, expression_list, color_label = "bulk_labels", vmin=-2, vmax=2, palette=sns.color_palette("tab10"), save_directory=None):   #sc_palette
    
    names = expression_list + [color_label]
    n = np.int(np.ceil(np.sqrt(len(names)+1)))
    dot_size = 50

    path = os.getcwd()

    if save_directory is not None:
        path = path + save_directory

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure(layout='constrained', figsize=(20, 16))

    fig.set_facecolor('lightgray')
    fig.suptitle(save_directory)

    subfigs = fig.subfigures(1, 2, wspace=0.01)
    subfigs[0].set_facecolor('teal')
    subfigs[1].set_facecolor('coral')


    subfigsnestL = subfigs[0].subfigures(2, 1)#, height_ratios=[1, 1.4])

    if "X" in rep:

        subfigsnestL[0].suptitle('UMAP of the original data')
        subfigsnestL[0].set_facecolor('w')
        axsnestX = subfigsnestL[0].subplots(n, n)

        sc.pp.neighbors(adata, use_rep = "X")
        sc.tl.umap(adata, n_components=2)

        for i, axs in enumerate(axsnestX):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=vmin, vmax=vmax)

                elif i * n + j == len(names)-1:
                    sc.pl.umap(adata,color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, legend_loc=None, palette=palette)

                elif i * n + j == len(names):
                    l1=ax.legend(handles=[
                            # Instead of Line2D we can also use other matplotlib objects, such as Patch, etc.
                            Line2D([0], [0], marker='o', color=c,lw=0,
                            label=l, markerfacecolor=c, markersize=5)
                            # Color groups in adata
                            for l,c in zip(
                                list(adata.obs[names[i * n + j - 1]].cat.categories),
                                palette)
                            ],prop={'size': 7},frameon=False,bbox_to_anchor=(1,1.2),title='Cell type')
                    ax.axis('off')  # Turn off the axis to display only the legend

                else:
                    # If there are no more names, remove the empty axes
                    subfigsnestL[0].delaxes(ax)
        #subfigsnest[0].colorbar(pc, ax=axsnest0)


    if "X_svgsa" in rep:

        subfigsnestL[1].suptitle('UMAP embedding of both the latent layers combined')
        subfigsnestL[1].set_facecolor('w')
        axsnestX_svgsa = subfigsnestL[1].subplots(n, n)

        sc.pp.neighbors(adata, use_rep = "X_svgsa")
        sc.tl.umap(adata, n_components=2)

        for i, axs in enumerate(axsnestX_svgsa):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=vmin, vmax=vmax)

                elif i * n + j == len(names)-1:
                    sc.pl.umap(adata,color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, legend_loc=None, palette=palette)

                elif i * n + j == len(names):
                    l1=ax.legend(handles=[
                            # Instead of Line2D we can also use other matplotlib objects, such as Patch, etc.
                            Line2D([0], [0], marker='o', color=c,lw=0,
                            label=l, markerfacecolor=c, markersize=5)
                            # Color groups in adata
                            for l,c in zip(
                                list(adata.obs[names[i * n + j - 1]].cat.categories),
                                palette)
                            ],prop={'size': 7},frameon=False,bbox_to_anchor=(1,1.2),title='Cell type')
                    ax.axis('off')  # Turn off the axis to display only the legend

                else:
                    # If there are no more names, remove the empty axes
                    subfigsnestL[1].delaxes(ax)


    subfigsnestR = subfigs[1].subfigures(2, 1)#, height_ratios=[1, 1.4])


    if "X_svgsa_gs" in rep:

        subfigsnestR[0].suptitle('UMAP embedding of the latent layer gs')
        subfigsnestR[0].set_facecolor('w')
        axsnestGS = subfigsnestR[0].subplots(n, n)

        sc.pp.neighbors(adata, use_rep = "X_svgsa_gs")
        sc.tl.umap(adata, n_components=2)

        for i, axs in enumerate(axsnestGS):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=vmin, vmax=vmax)

                elif i * n + j == len(names)-1:
                    sc.pl.umap(adata,color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, legend_loc=None, palette=palette)

                elif i * n + j == len(names):
                    l1=ax.legend(handles=[
                            # Instead of Line2D we can also use other matplotlib objects, such as Patch, etc.
                            Line2D([0], [0], marker='o', color=c,lw=0,
                            label=l, markerfacecolor=c, markersize=5)
                            # Color groups in adata
                            for l,c in zip(
                                list(adata.obs[names[i * n + j - 1]].cat.categories),
                                palette)
                            ],prop={'size': 7},frameon=False,bbox_to_anchor=(1,1.2),title='Cell type')
                    ax.axis('off')  # Turn off the axis to display only the legend

                else:
                    # If there are no more names, remove the empty axes
                    subfigsnestR[0].delaxes(ax)


    if "X_svgsa_uns" in rep:

        subfigsnestR[1].suptitle('UMAP embedding of the latent layer uns')
        subfigsnestR[1].set_facecolor('w')
        axsnestUNS = subfigsnestR[1].subplots(n, n)

        sc.pp.neighbors(adata, use_rep = "X_svgsa_uns")
        sc.tl.umap(adata, n_components=2)

        for i, axs in enumerate(axsnestUNS):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=vmin, vmax=vmax)

                elif i * n + j == len(names)-1:
                    sc.pl.umap(adata,color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, legend_loc=None, palette=palette)

                elif i * n + j == len(names):
                    l1=ax.legend(handles=[
                            # Instead of Line2D we can also use other matplotlib objects, such as Patch, etc.
                            Line2D([0], [0], marker='o', color=c,lw=0,
                            label=l, markerfacecolor=c, markersize=5)
                            # Color groups in adata
                            for l,c in zip(
                                list(adata.obs[names[i * n + j - 1]].cat.categories),
                                palette)
                            ],prop={'size': 7},frameon=False,bbox_to_anchor=(1,1.2),title='Cell type')
                    ax.axis('off')  # Turn off the axis to display only the legend

                else:
                    # If there are no more names, remove the empty axes
                    subfigsnestR[1].delaxes(ax)


    plt.savefig(os.path.join(path,'full_plot.png'), bbox_inches="tight")