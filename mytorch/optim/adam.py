import numpy as np

class Adam():
    """Adam optimizer
    """
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Initialize optimizer
        Args:
            model (Model): model to optimize
            lr (float): learning rate
            beta1 (float): beta1 parameter for Adam
            beta2 (float): beta2 parameter for Adam
            eps (float): epsilon parameter for Adam
        """
        self.l = model.layers
        self.L = len(model.layers)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in model.layers]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in model.layers]

    def step(self):
        """Performs one optimization step
        """
        self.t += 1 # increment timestep
        for layer_id, layer in enumerate(self.l):
            # Calculate updates for weight
            self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1-self.beta1) * layer.dLdW
            self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1-self.beta2) * (layer.dLdW ** 2)
            m_W_hat = self.m_W[layer_id] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[layer_id] / (1 - self.beta2 ** self.t)
            
            # calculate updates for bias
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1-self.beta1) * layer.dLdb
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1-self.beta2) * (layer.dLdb ** 2)
            m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** self.t)

            # Perform weight and bias updates
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)