import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        x = np.array(x)
        # w: 1D weight array
        w = np.array(w)
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x,w) + b
        z_clipped = np.clip(z,-500,500)
        y_pred = 1/(1+np.exp(-z_clipped))

        # Loss: L = 0.5 * (y_hat - y_true)^2
        l = 0.5 * (y_pred - y_true)**2
        dl_dw = (y_pred - y_true)*(y_pred*(1-y_pred))*x
        dl_db = (y_pred - y_true)*(y_pred*(1-y_pred))
        dl_dw = np.round(dl_dw,5)
        dl_db = np.round(dl_db,5)
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return dl_dw,dl_db