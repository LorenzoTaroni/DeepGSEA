import sklearn as sk
import numpy as np
from sklearn.preprocessing import scale


def select_highest_variance_gs(X, gs, N, gene_names, normalize = True):
    if normalize:
        Z = scale(np.log1p(X.transpose() / X.sum(axis=1) * 10 ** 6), axis=1, with_mean=True,
                  with_std=False, copy=True)
    else:
        Z = X.transpose()
    
    ks = list(gs.keys())

    vars_gs = np.array([calc_var(Z,gs[k][0], gene_names) for k in ks]) 

    idx = (-vars_gs).argsort()[:N]
    return np.array(ks)[idx]

def calc_var(Z, gene_set, gene_names):

    idx = np.intersect1d(np.array(gene_names), np.array(gene_set), return_indices=True)[1]
    Z = Z[idx,:]
    var = np.var(Z, axis = 1)
    return(var.sum() / len(gene_set) )