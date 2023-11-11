import numpy as np

class MSELoss:
    """Mean Squared Error Loss
    """
    def forward(self, A, Y):
        """Calculates mean squared error loss given model output and ground truth values.
        Args:
            A (np.ndarray): Output of the model of shape (N, C).
            Y (np.ndarray): Ground-truth values of shape (N, C).
        Returns:
            float: MSE Loss.
        """
        self.A, self.Y = A, Y
        self.N, self.C = A.shape[0], A.shape[1]
        mse = np.sum((A - Y) ** 2) / (self.N * self.C)
        return mse

    def backward(self):
        """Calculates the gradient of the MSE loss with respect to the model output.
        Returns:
            np.ndarray: Gradient of the MSE loss with respect to the model output.
        """
        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)
        return dLdA
    

class CrossEntropyLoss:
    """Cross Entropy Loss
    """
    def forward(self, A, Y):
        """Calculates cross entropy loss given model output and ground truth values.
        Args:
            A (np.ndarray): Output of the model of shape (N, C).
            Y (np.ndarray): Ground-truth values of shape (N, C).
        Returns:
            float: Cross Entropy Loss.
        """
        self.A, self.Y = A, Y
        self.N, self.C = A.shape[0], A.shape[1]
        self.softmax = np.exp(A-np.max(A)) / np.exp(A-np.max(A)).sum(axis=1, keepdims=True)
        crossentropy = (-Y * np.log(self.softmax)).sum(axis=1)
        L = crossentropy.sum() / self.N
        return L

    def backward(self):
        """Calculates the gradient of the cross entropy loss with respect to the model output.
        Returns:
            np.ndarray: Gradient of the cross entropy loss with respect to the model output.
        """
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA