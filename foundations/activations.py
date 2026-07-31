import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # sig = 1/(1 + np.exp(-z))
        # return np.round(your_answer, 5)
        z = np.array(z)
        return np.round(1 / (1 + np.exp(-z)), 5)
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        z = np.array(z)
        rel = np.maximum(0,z)
        return rel
z = [-1.0, 0.0, 1.0, 2.0]
print(Solution.sigmoid(0,z))
print(Solution.relu(0,z))