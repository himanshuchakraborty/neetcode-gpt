import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        x = np.array(x)
        # w: 1D weight array (same length as x)
        w = np.array(w)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        # activation = "sigmoid"
        z = np.dot(x,w) + b

        if activation == "relu":
        #
            # Pre-activation: z = dot(x, w) + b
            # ReLU: max(0, z)
            relu = np.maximum(0,z)
            return np.round(relu,5)

        elif activation == "sigmoid":
            # Sigmoid: σ(z) = 1 / (1 + exp(-z))
            z_clipped = np.clip(z,-500,500)
            sigmoid = 1 /(1 + np.exp(-z_clipped))
            return np.round(sigmoid,5)

 
        # return round(your_answer, 5)
        # return np.round(sigmoid,5)
        # return np.round(relu,5)
        
