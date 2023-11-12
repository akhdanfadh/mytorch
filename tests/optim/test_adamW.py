import unittest
import os
import pickle
import numpy as np
from mytorch.models.mlp import MLP4
from mytorch.optim.adamW import AdamW

pkl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')

class TestAdam(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.lr = 0.008
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.weight_decay = 0.01
        self.n_step = 5

        self.A0 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]], dtype="f")
        self.dLdA5 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]], dtype="f")

    def mlp4_forward(self, mlp4):
        with open(f"{pkl_dir}/mlp4_W.pkl", "rb") as f:
            Ws = pickle.load(f)
        with open(f"{pkl_dir}/mlp4_b.pkl", "rb") as f:
            bs = pickle.load(f)
        for i in range(len(mlp4.layers)):
            mlp4.layers[i].W = Ws[i]
            mlp4.layers[i].b = bs[i]
        A5 = mlp4.forward(self.A0)
        return mlp4
    
    def mlp4_backward(self, mlp4):
        for i in range(len(mlp4.layers)):
            mlp4.layers[i].dLdW.fill(0.0)
            mlp4.layers[i].dLdb.fill(0.0)
        mlp4.backward(self.dLdA5)

    def test_adam_step(self):
        mlp4 = MLP4(debug=True)
        optimizer = AdamW(mlp4,
                          lr=self.lr,
                          beta1=self.beta1,
                          beta2=self.beta2,
                          eps=self.eps,
                          weight_decay=self.weight_decay)

        # Forward and backward n_step times
        for _ in range(self.n_step):
            self.mlp4_forward(mlp4)
            self.mlp4_backward(mlp4)
            optimizer.step()

        # Student outputs
        get_W = [x.W.round(4) for x in optimizer.l]
        get_b = [x.b.round(4) for x in optimizer.l]

        # Expected outputs
        with open(f"{pkl_dir}/adamW_sol_W.pkl", "rb") as f:
            sol_W = pickle.load(f)
        with open(f"{pkl_dir}/adamW_sol_b.pkl", "rb") as f:
            sol_b = pickle.load(f)

        # Assertions
        for get, sol in zip(get_W, sol_W):
            self.assertTrue(np.allclose(get, sol, atol=self.atol_threshold))
        for get, sol in zip(get_b, sol_b):
            self.assertTrue(np.allclose(get, sol, atol=self.atol_threshold))


if __name__ == '__main__':
    unittest.main()