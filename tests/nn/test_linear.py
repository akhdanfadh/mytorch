import unittest
import numpy as np
from mytorch.nn.linear import Linear

class TestLinear(unittest.TestCase):
    def setUp(self):
        # Set up the data and expected results for testing
        self.A = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.W = np.array([[-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.b = np.array([[-1.], [0.], [1.]], dtype="f")

        self.Z_solution = np.array([[10., -3., -16.], [4., -1., -6.], [-2., 1., 4.], [-8., 3., 14.]], dtype="f")
        self.dLdA_solution = np.array([[4., -5.], [4., 4.], [4., 13.], [4., 22.]], dtype="f")
        self.dLdW_solution = np.array([[28., 30.], [24., 30.], [20., 30.]], dtype="f")
        self.dLdb_solution = np.array([[2.], [6.], [10.]], dtype="f")

        self.atol_threshold = 1e-4

    def test_forward(self):
        linear = Linear(2, 3, debug=True)
        linear.W = self.W
        linear.b = self.b

        Z = linear.forward(self.A)
        np.testing.assert_allclose(Z, self.Z_solution, atol=self.atol_threshold)

    def test_backward(self):
        linear = Linear(2, 3, debug=True)
        linear.W = self.W
        linear.b = self.b

        dLdZ = np.array([[-4., -3., -2.], [-1., 0., 1.], [2., 3., 4.], [5., 6., 7.]], dtype="f")
        linear.forward(self.A)
        linear.backward(dLdZ)

        np.testing.assert_allclose(linear.dLdA, self.dLdA_solution, atol=self.atol_threshold)
        np.testing.assert_allclose(linear.dLdW, self.dLdW_solution, atol=self.atol_threshold)
        np.testing.assert_allclose(linear.dLdb, self.dLdb_solution, atol=self.atol_threshold)

if __name__ == '__main__':
    unittest.main()
