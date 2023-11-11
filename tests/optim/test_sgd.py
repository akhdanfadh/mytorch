import unittest
import numpy as np
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU
from mytorch.optim.sgd import SGD

class TestSGD(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.lr = 0.9

    def test_sgd_step(self):
        class PseudoModel:
            def __init__(self):
                self.layers = [Linear(3, 2)]
                self.f = [ReLU()]

            def forward(self, A):
                return NotImplemented

            def backward(self):
                return NotImplemented

        # Initialize model and optimizer
        pseudo_model = PseudoModel()
        pseudo_model.layers[0].W = np.ones((3, 2))
        pseudo_model.layers[0].dLdW = np.ones((3, 2)) / 10
        pseudo_model.layers[0].b = np.ones((3, 1))
        pseudo_model.layers[0].dLdb = np.ones((3, 1)) / 10
        optimizer = SGD(pseudo_model, lr=self.lr)

        # Expected solutions
        W_1_solution = np.array([[0.91, 0.91], [0.91, 0.91], [0.91, 0.91]], dtype="f")
        b_1_solution = np.array([[0.91], [0.91], [0.91]], dtype="f")
        W_2_solution = np.array([[0.82, 0.82], [0.82, 0.82], [0.82, 0.82]], dtype="f")
        b_2_solution = np.array([[0.82], [0.82], [0.82]], dtype="f")

        # First SGD step
        optimizer.step()
        W_1 = pseudo_model.layers[0].W
        b_1 = pseudo_model.layers[0].b
        self.assertTrue(np.allclose(W_1, W_1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(b_1, b_1_solution, atol=self.atol_threshold))

        # Second SGD step
        optimizer.step()
        W_2 = pseudo_model.layers[0].W
        b_2 = pseudo_model.layers[0].b
        self.assertTrue(np.allclose(W_2, W_2_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(b_2, b_2_solution, atol=self.atol_threshold))

if __name__ == '__main__':
    unittest.main()
