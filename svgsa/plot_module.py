import os
import numpy as np
import scanpy as sc
import decoupler as dc
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from matplotlib.lines import Line2D
import seaborn as sns

def save_plot(file_name, save_directory):
    path = os.getcwd()

    if save_directory is not None:
        path = path + save_directory

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path,file_name+'.png'), bbox_inches="tight")


def decoupler_comparison(N_GS_list, adata, net, source='source', target='target', weight='weight', use_raw=True, method_list=None, save_directory=None):

    dc.decouple(mat=adata,net=net,source=source,target=target,weight=weight, methods=method_list, use_raw=use_raw)

    # Make a list of formatted method names
    estimate_list = []
    for name in method_list:
        estimate_list.append(name + '_estimate')

    # Define the number of plots
    num_plots = len(N_GS_list)

    # Create the figure and subplots
    fig, axes = plt.subplots(nrows=len(estimate_list), ncols=num_plots, figsize=(4*num_plots, 4*len(estimate_list)))
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    
    for i, axs in enumerate(axes):
        method = estimate_list[i]
        #adata_new.obsm["mlm_estimate"][adata_new.obsm["mlm_estimate"].keys()[i]]
        # Iterate over each value of i and create a scatter plot
        for j, ax in enumerate(axs):
            sns.scatterplot(x=adata.obsm[method][N_GS_list[j]], y=adata.obsm["last_node"][N_GS_list[j]],
                            hue=adata.obs[N_GS_list[j]], palette="viridis", ax=ax, legend=False) #, palette=f"tab{palette_value}"
            
            #g.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), ncol=1)

            norm = plt.Normalize(adata.obs[N_GS_list[j]].min(), adata.obs[N_GS_list[j]].max())
            cmap = sns.color_palette("viridis", as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cax = fig.add_axes([ax.get_position().x1 + 0.001, ax.get_position().y0, ax.get_position().width / 20, ax.get_position().height / 2])
            ax.figure.colorbar(sm, cax=cax)

            ax.set_xlabel(f'{N_GS_list[j]} {method}')
            ax.set_ylabel(f'Encoder_GSEA ending node score {j+1}')

    save_plot("decoupler_comparison", save_directory)

def plot_loss(loss, save_directory=None):

    plt.figure(figsize=(5, 2))
    plt.plot(loss)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    save_plot('ElboLoss', save_directory)


def plot_learning_rate_decay(lrs, save_directory=None):

    plt.figure(figsize=(5, 2))
    plt.plot(lrs)
    plt.xlabel("SVI step")
    plt.ylabel("lr decay")
    save_plot('lr_decay', save_directory)


def umap_embedding(adata, rep, expression_list, color_label = "bulk_labels", ncols=4, vmin=-2, vmax=2, palette=None, save_directory=None):   #sc_palette

    # umap hyperparameters
    random_state=42
    n_neighbors=15
    min_dist=0.5
    
    sc.pp.neighbors(adata, use_rep = rep, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    with rc_context({'figure.figsize': (3, 3)}):
        if palette:
            sc.pl.umap(adata, color=expression_list + [color_label], s=50, frameon=False, ncols=ncols, vmin=vmin, vmax=vmax, palette=palette, show=False)
        else:
            sc.pl.umap(adata, color=expression_list + [color_label], s=50, frameon=False, ncols=ncols, vmin=vmin, vmax=vmax, show=False)
        save_plot(rep, save_directory)


def umap_full_plot(adata, rep, expression_list, color_label = "bulk_labels", dot_size=50, palette=sns.color_palette("tab10"), save_directory=None, seed = None):   #sc_palette
    
    names = expression_list + [color_label]
    n = int(np.ceil(np.sqrt(len(names)+1)))

    # umap hyperparameters
    n_neighbors=50
    min_dist=0.6
    random_state=42

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

        sc.pp.neighbors(adata, use_rep = "X", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        for i, axs in enumerate(axsnestX):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=adata.obs[names[i * n + j]].min(), vmax=adata.obs[names[i * n + j]].max())

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

        sc.pp.neighbors(adata, use_rep = "X_svgsa", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        for i, axs in enumerate(axsnestX_svgsa):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=adata.obs[names[i * n + j]].min(), vmax=adata.obs[names[i * n + j]].max())

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

        sc.pp.neighbors(adata, use_rep = "X_svgsa_gs", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        for i, axs in enumerate(axsnestGS):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=adata.obs[names[i * n + j]].min(), vmax=adata.obs[names[i * n + j]].max())

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

        sc.pp.neighbors(adata, use_rep = "X_svgsa_uns", n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

        for i, axs in enumerate(axsnestUNS):
            for j, ax in enumerate(axs):
                if i * n + j < (len(names)-1):
                    # Use names[i * n + j] to set the color or other parameters for the plot
                    sc.pl.umap(adata, color=names[i * n + j], ax=ax, s=dot_size, frameon=False, show=False, vmin=adata.obs[names[i * n + j]].min(), vmax=adata.obs[names[i * n + j]].max())

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

    if seed is not None:
        save_plot(f'seed={seed}_umap_full_plot', save_directory)
    else:
        save_plot('umap_full_plot', save_directory)