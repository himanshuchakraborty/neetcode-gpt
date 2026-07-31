import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        z = np.array(z)
        # Hint: subtract max(z) for numerical stability before computing exp
        z_clip = z - np.max(z)
        soft = np.exp(z_clip)/(np.sum(np.exp(z_clip)))
        # return np.round(your_answer, 4)
        return np.round(soft,4)
