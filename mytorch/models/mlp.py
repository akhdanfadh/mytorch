import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

class MLP0:
    """MLP with a single linear layer of shape (2,3) and ReLU activation
    """
    def __init__(self, debug=False):
        self.layers = [Linear(2, 3)]
        self.f = [ReLU()]
        self.debug = debug

    def forward(self, A0):
        """Forward pass: compute predicted y given x
        Args:
            A0 (array-like): input data of shape (batch_size, 2)
        Returns:
            A1 (array-like): output data of shape (batch_size, 3)
        """
        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0)

        if self.debug: self.Z0, self.A1 = Z0, A1
        return A1
    
    def backward(self, dLdA1):
        """Backward pass: compute gradient w.r.t. input and gradient w.r.t. weight and bias given output gradient
        Args:
            dLdA1 (array-like): gradient w.r.t. output of shape (batch_size, 3)
        Returns:
            dLdA0 (array-like): gradient w.r.t. input of shape (batch_size, 2)
        """
        dLdZ0 = self.f[0].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug: self.dLdZ0, self.dLdA0 = dLdZ0, dLdA0
        return dLdA0


class MLP1:
    """MLP with 2 linear layers each with ReLU activation
    Layer 1: Linear(2, 3)
    Layer 2: Linear(3, 2)
    """
    def __init__(self, debug=False):
        self.layers = [Linear(2, 3), Linear(3, 2)]
        self.f = [ReLU(), ReLU()]
        self.debug = debug
    
    def forward(self, A0):
        """Forward pass: compute predicted y given x
        Args:
            A0 (array-like): input data of shape (batch_size, 2)
        Returns:
            A2 (array-like): output data of shape (batch_size, 2)
        """
        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0)
        Z1 = self.layers[1].forward(A1)
        A2 = self.f[1].forward(Z1)

        if self.debug: self.Z0, self.A1, self.Z1, self.A2 = Z0, A1, Z1, A2
        return A2

    def backward(self, dLdA2):
        """Backward pass: compute gradient w.r.t. input and gradient w.r.t. weight and bias given output gradient
        Args:
            dLdA2 (array-like): gradient w.r.t. output of shape (batch_size, 2)
        Returns:
            dLdA0 (array-like): gradient w.r.t. input of shape (batch_size, 2)
        """
        dLdZ1 = self.f[1].backward(dLdA2)
        dLdA1 = self.layers[1].backward(dLdZ1)
        dLdZ0 = self.f[0].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:  
            self.dLdZ1, self.dLdA1 = dLdZ1, dLdA1
            self.dLdZ0, self.dLdA0 = dLdZ0, dLdA0
        return dLdA0


class MLP4:
    """MLP with 4 hidden layers and an output layer, each with ReLU activation

    Layer 1: Linear(2, 4)
    Layer 2: Linear(4, 8)
    Layer 3: Linear(8, 8)
    Layer 4: Linear(8, 4)
    Output Layer: Linear(4, 2)
    """
    def __init__(self, debug=False):
        self.layers = [
            Linear(2, 4),
            Linear(4, 8),
            Linear(8, 8),
            Linear(8, 4),
            Linear(4, 2),
        ]
        self.f = [ReLU() for _ in range(5)]
        self.debug = debug
    
    def forward(self, A):
        """Forward pass: compute predicted y given x
        Args:
            A (array-like): input data of shape (batch_size, 2)
        Returns:
            A (array-like): output data of shape (batch_size, 2)
        """
        if self.debug: self.Z, self.A = [], [A]
        for i in range(len(self.layers)):
            Z = self.layers[i].forward(A)
            A = self.f[i].forward(Z)
            if self.debug: self.Z.append(Z), self.A.append(A)
        return A
    
    def backward(self, dLdA):
        """Backward pass: compute gradient w.r.t. input and gradient w.r.t. weight and bias given output gradient
        Args:
            dLdA (array-like): gradient w.r.t. output of shape (batch_size, 2)
        Returns:
            dLdA (array-like): gradient w.r.t. input of shape (batch_size, 2)
        """
        if self.debug: self.dLdZ, self.dLdA = [], [dLdA]
        for i in reversed(range(len(self.layers))):
            dLdZ = self.f[i].backward(dLdA)
            dLdA = self.layers[i].backward(dLdZ)
            if self.debug: # order matters, prepend
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA
        return dLdA