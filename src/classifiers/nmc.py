import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        """
        Estimate the centroid for each class from the training data.

        Parameters
        ----------
        xtr : ndarray
            the training data matrix, with one sample per row.

        ytr : ndarray
            the labels of the training samples.

        Returns
        -------
        self
        """
        self._class_labels = np.unique(ytr)
        n_classes = self._class_labels.size
        n_features = xtr.shape[1]

        centroids = np.zeros(shape=(n_classes, n_features))
        for i, label in enumerate(self._class_labels):
            centroids[i, :] = xtr[ytr == label, :].mean(axis=0)

        self._centroids = centroids
        return self

    def predict(self, xts):
        """
        Predict the class label of each testing sample, based on the
        closest centroid estimated during training.

        Parameters
        ----------
        xts : ndarray
            the testing data matrix, with one sample per row.

        Returns
        -------
        ypred : ndarray
            the predicted labels for each testing sample.
        """
        if self._centroids is None:
            raise ValueError("The classifier has not been trained. "
                              "Call fit(x, y) before predict(x).")

        dist = euclidean_distances(xts, self._centroids)
        idx_min = np.argmin(dist, axis=1)
        ypred = self._class_labels[idx_min]
        return ypred
