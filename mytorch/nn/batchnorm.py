import numpy as np

class BatchNorm1d:
    """Batch Normalization layer.
    """
    def __init__(self, num_features, alpha=0.9):
        """Initialize the layer.
        Args:
            num_features (int): Number of features in input.
            alpha (float): Smoothing parameter.
        """
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))
    
    def forward(self, Z, eval=False):
        """Forward pass.
        Args:
            Z (np.array): Input array of shape (N, num_features).
            eval (bool): Training or inference (whether to use running mean and variance).
        Returns:
            np.array: Output array of shape (N, num_features).
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = (Z.sum(axis=0, keepdims=True)) / self.N                # mean
        self.V = ((Z - self.M)**2).sum(axis=0, keepdims=True) / self.N  # variance

        if eval == False: # training mode
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)         # normalized Z
            self.BZ = self.BW * self.NZ + self.Bb                       # batch normalized Z

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else: # inference mode
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb
        
        return self.BZ

    def backward(self, dLdBZ):
        """Backward pass.
        Args:
            dLdBZ (np.array): Gradient of loss with respect to output array, shape (N, num_features).
        Returns:
            np.array: Gradient of loss with respect to input array, shape (N, num_features).
        """
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW

        inv_sqrt_V = 1 / np.sqrt(self.V + self.eps)
        Z_min_M = self.Z - self.M
        dLdV = -0.5 * np.sum(dLdNZ * Z_min_M * (inv_sqrt_V ** 3), axis=0, keepdims=True)

        dNZdM = -inv_sqrt_V - (0.5 * Z_min_M) * (inv_sqrt_V ** 3) * \
              (-2 / self.N * np.sum(Z_min_M, axis=0, keepdims=True))
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True)

        dLdZ = dLdNZ * (1 / np.sqrt(self.V + self.eps)) + \
               dLdV * (2 / self.N * Z_min_M) + 1 / self.N * dLdM

        return dLdZ