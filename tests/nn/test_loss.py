import unittest
import numpy as np
from mytorch.nn.loss import MSELoss, CrossEntropyLoss

class TestMSELoss(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4

    def test_mseloss_forward_backward(self):
        A = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        Y = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]], dtype="f")
        L_solution = np.array(6.5, dtype="f")
        dLdA_solution = np.array([[-0.5, -0.5], [-0.375, -0.125], [-0.125, 0.125], [0.25, 0.25]], dtype="f") * 2

        mse = MSELoss()

        # Forward pass
        L = mse.forward(A, Y)
        self.assertTrue(np.allclose(L.round(4), L_solution, atol=self.atol_threshold))

        # Backward pass
        dLdA = mse.backward()
        self.assertTrue(np.allclose(dLdA.round(4), dLdA_solution, atol=self.atol_threshold))


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4

    def test_crossentropyloss_forward_backward(self):
        A = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        Y = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]], dtype="f")
        L_solution = np.array(0.8133, dtype="f")
        dLdA_solution = np.array([[0.2689, -0.2689], [-0.7311, 0.7311], [-0.7311, 0.7311], [0.2689, -0.2689]], dtype="f") / 4

        xent = CrossEntropyLoss()

        # Forward pass
        L = xent.forward(A, Y)
        self.assertTrue(np.allclose(L.round(4), L_solution, atol=self.atol_threshold))

        # Backward pass
        dLdA = xent.backward()
        self.assertTrue(np.allclose(dLdA.round(4), dLdA_solution, atol=self.atol_threshold))


if __name__ == '__main__':
    unittest.main()
