# based on the Github code of Van den Berg https://github.com/riannevdberg/gc-mc
#modified by Juliette Valenchon

from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import pandas as pd


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')
    v_features = sp.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features


"""def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    # Globally Normalizes set of bipartite adjacency matrices 

    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I

    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt= []
    degree_v_inv_sqrt=[]
    for i in range(degree_u.shape[0]):
        if degree_u[i] <0:
            degree_u_inv_sqrt.append(- 1. / np.sqrt(-degree_u[i]))
        else:
            degree_u_inv_sqrt.append(1. / np.sqrt(degree_u[i]))

    for i in range(degree_v.shape[0]):
        if degree_v[i] <0:
            degree_v_inv_sqrt.append(-1. / np.sqrt(-degree_v[i]))
        else:

            degree_v_inv_sqrt.append(1. / np.sqrt(degree_v[i]))
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm"""


"""def globally_normalize_bipartite_adjacency_absolute(adjacencies, verbose=False, symmetric=True):
    # Globally Normalizes (absolute norm) set of bipartite adjacency matrices 
    # degree_u and degree_v are row and column sums of adj+I
    adj_tot = np.sum(np.absolute(adj) for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf
    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])


    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)

    if symmetric:
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]

    return adj_norm"""

def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_data_monti_tadpole(M, Otraining, Otest, Ovalidation, testing=False):
    """
    Loads TADPOLE data
    Inputs:
        M : normalized matrix (between -1 and 1, except for the labels) with -10 for missing values
        Otraining, Otest, Ovalidation: mask on M to know which elements are in which set
        testing: if False then we have a validation set. Otherwise, we concatenate the training and validation set.
    """

    num_subjects = M.shape[0]
    num_features = M.shape[1]

    #no side information
    u_features = sp.identity(num_subjects, format='csr')
    v_features = sp.identity(num_features, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

    labels = M.reshape([-1])
    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = np.where(Ovalidation)[0].shape[0]

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_features + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_features + v for u, v in pairs_nonzero_test])

    pairs_nonzero_val = np.array([[u, v] for u, v in zip(np.where(Ovalidation)[0], np.where(Ovalidation)[1])])
    idx_nonzero_val = np.array([u * num_features + v for u, v in pairs_nonzero_val])


    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_val, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_val, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero_val
    train_idx = idx_nonzero_train

    test_idx = idx_nonzero_test

    assert(len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero_val
    train_pairs_idx = pairs_nonzero_train
    test_pairs_idx = pairs_nonzero_test

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]


    if testing:
        print('stack test val')
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])
    return u_features, v_features, train_labels, u_train_idx, v_train_idx, \
        val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx
