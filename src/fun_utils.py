from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two disjoint random subsets

    Parameters
    ----------
    x : ndarray
        the data matrix, with one sample per row.

    y : ndarray
        the labels of each sample.

    tr_fraction : float, optional
        the fraction of samples assigned to the training set (default 0.5).

    Returns
    -------
    xtr, ytr : ndarray
        samples and labels of the training set.

    xts, yts : ndarray
        samples and labels of the testing set.
    """
    n_samples = x.shape[0]
    n_tr = int(n_samples * tr_fraction)

    # random permutation of the sample indices, then split it in two
    # non-overlapping parts, so that training and testing sets are disjoint
    idx = np.random.permutation(n_samples)
    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    xtr = x[tr_idx, :]
    ytr = y[tr_idx]
    xts = x[ts_idx, :]
    yts = y[ts_idx]

    return xtr, ytr, xts, yts
