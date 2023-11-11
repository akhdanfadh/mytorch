import numpy as np
from scipy.special import erf

class Identity:
    """Identity activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = dLdA * np.ones(self.A.shape, dtype="f")
        return dLdZ


class Sigmoid:
    """Sigmoid activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = dLdA * self.A * (1 - self.A)
        return dLdZ


class Tanh:
    """Tanh activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = dLdA * (1 - self.A ** 2)
        return dLdZ


class ReLU:
    """ReLU activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        self.A = np.maximum(0, Z)
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = dLdA * (self.A > 0)
        return dLdZ


class GELU:
    """GELU activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        self.Z = Z
        self.A = 0.5 * Z * (1 + erf(Z / np.sqrt(2)))
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = dLdA * (0.5 * (1 + erf(self.Z / np.sqrt(2))) + \
                self.Z * np.exp(-self.Z ** 2 / 2) / np.sqrt(2 * np.pi))
        return dLdZ


class Softmax:
    """Softmax activation function class
    """
    def __init__(self):
        self.A = None

    def forward(self, Z):
        """Forward pass: compute output given input
        Args:
            Z (array-like): input data of shape (batch_size, in_features)
        Returns:
            A (array-like): output data of shape (batch_size, in_features)
        """
        exp_X = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # for numerical stability
        self.A = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, in_features)
        Returns:
            dLdZ (array-like): gradient w.r.t. input of shape (batch_size, in_features)
        """
        dLdZ = self.A * (dLdA - np.sum(self.A * dLdA, axis=1, keepdims=True))
        return dLdZ