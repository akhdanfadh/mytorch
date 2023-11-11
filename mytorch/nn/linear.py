import numpy as np

class Linear():
    """Linear layer also known as fully connected layer
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
    """
    def __init__(self, in_features, out_features, debug=False):
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))
        self.dLdW = np.zeros((out_features, in_features))
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug

    def forward(self, A):
        """Forward pass: compute predicted output given input
        Args:
            A (array-like): input data of shape (batch_size, in_features)
        Returns:
            Z (array-like): output data of shape (batch_size, out_features)
        """
        self.A = np.array(A)
        self.Ones = np.ones((A.shape[0], 1))
        self.Z = A @ self.W.T + self.Ones @ self.b.T
        return self.Z

    def backward(self, dLdZ):
        """Backward pass: compute gradient w.r.t. input and gradient w.r.t. weight and bias given output gradient
        Args:
            dLdZ (array-like): gradient w.r.t. output of shape (batch_size, out_features)
        Returns:
            dLdA (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.Ones
        dLdA = dLdZ @ self.W

        if self.debug: self.dLdA = dLdA
        return dLdA