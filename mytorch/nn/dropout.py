import numpy as np

class Dropout(object):
    """Dropout layer.
    """
    def __init__(self, p=0.5):
        """Initialize the layer.
        Args:
            p (float): Probability of dropping a neuron.
        """
        self.p = p

    def __call__(self, x):
        """Call the layer.
        Args:
            x (np.array): Input array of shape (N, num_features).
        """
        return self.forward(x)

    def forward(self, x, train=True):
        """Forward pass.
        As dropout zeros out a portion of the tensor, we need to re-scale the remaining numbers
        so the total 'intensity' of the output is same as in testing, where we don't apply dropout.

        If the network learned to rely on the reduced output (because of dropout during training),
        directly using all neurons during testing can lead to an output that's scaled up, potentially
        disrupting the network's ability to make accurate predictions.

        Args:
            x (np.array): Input array of shape (N, num_features).
            train (bool): Whether to use the layer in training mode or in inference mode.
        """
        if train:
            self.mask = np.logical_not(np.random.binomial(1, self.p, size=x.shape))
            return x * self.mask / (1 - self.p)
        else:
            return x
		
    def backward(self, delta):
        """Backward pass.
        Args:
            delta (np.array): Upstream derivative, of shape (N, num_features).
        """
        return delta * self.mask
