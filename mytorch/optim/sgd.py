import numpy as np

class SGD:
    """Minibatch Stochastic Gradient Descent optimizer with momentum.
    """
    def __init__(self, model, lr=0.1, momentum=0):
        """Initializes the SGD optimizer.
        Args:
            model (Model): Model to be optimized.
            lr (float): Learning rate.
            momentum (float): Momentum.
        """
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):
        """Performs one step (weight update) of SGD.
        """
        for i in range(self.L):
            if self.mu == 0:
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb
            else: # with momentum
                self.v_W[i] = self.mu * self.v_W[i] + self.l[i].dLdW
                self.v_b[i] = self.mu * self.v_b[i] + self.l[i].dLdb
                self.l[i].W -= self.lr * self.v_W[i]
                self.l[i].b -= self.lr * self.v_b[i]