import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_pred)
        # y_true: true labels (0 or 1)
        y_true = np.array(y_true)
        # y_pred: predicted probabilities
        y_pred = np.array(y_pred)
            # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        epsilon = 1e-7
        y_pred = y_pred + epsilon
        bce = -1/float(n)*np.sum((y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)))
        # return round(your_answer, 4)
        return np.round(bce , 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_pred)

        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        y_true = np.array(y_true)
    
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        y_pred = np.array(y_pred)
    
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        epsilon = 1e-7
        y_pred = y_pred + epsilon
        cce = -1/float(n)*np.sum((y_true*np.log(y_pred)))
    
        # return round(your_answer, 4)
        return np.round(cce , 4)
