import unittest
import numpy as np
from scipy.special import erf
from mytorch.nn.activation import Identity, Sigmoid, Tanh, ReLU, GELU, Softmax

class TestIdentity(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[-4, -3], [-2, -1], [0, 1], [2, 3]], dtype="f")
        self.dLdA = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.A_solution = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.dLdZ_solution = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.atol_threshold = 1e-4

    def test_identity_forward(self):
        identity = Identity()
        A = identity.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_identity_backward(self):
        identity = Identity()
        identity.forward(self.Z)
        dLdZ = identity.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[-4, -3], [-2, -1], [0, 1], [2, 3]], dtype="f")
        self.dLdA = np.array([[-4., -3.], [-2., -1.], [0., 1.], [2., 3.]], dtype="f")
        self.A_solution = np.array([[0.018, 0.0474], [0.1192, 0.2689], [0.5, 0.7311], [0.8808, 0.9526]], dtype="f")
        self.dLdZ_solution = np.array([[-0.0707, -0.1355], [-0.21, -0.1966], [0., 0.1966], [0.21, 0.1355]], dtype="f")
        self.atol_threshold = 1e-4

    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        A = sigmoid.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        sigmoid.forward(self.Z)
        dLdZ = sigmoid.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


class TestTanh(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[-4, -3], [-2, -1], [0, 1], [2, 3]], dtype="f")
        self.dLdA = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 0.0], [0.0, -1.0]], dtype="f")
        self.A_solution = np.array([[-0.9993, -0.9951], [-0.964, -0.7616], [0., 0.7616], [0.964, 0.9951]], dtype="f")
        self.dLdZ_solution = np.array([[1.300e-03, 9.900e-03], [2.121e-01, 4.200e-01], [2.000e+00, 0.000e+00], [0.000e+00, -9.900e-03]], dtype="f")
        self.atol_threshold = 1e-4

    def test_tanh_forward(self):
        tanh = Tanh()
        A = tanh.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_tanh_backward(self):
        tanh = Tanh()
        tanh.forward(self.Z)
        dLdZ = tanh.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[-4, -3], [-2, -1], [0, 1], [2, 3]], dtype="f")
        self.dLdA = np.array([[1.0, 1.0], [3.0, 1.0], [2.0, 0.0], [0.0, -1.0]], dtype="f")
        self.A_solution = np.array([[0., 0.], [0., 0.], [0., 1.], [2., 3.]], dtype="f")
        self.dLdZ_solution = np.array([[0., 0.], [0., 0.], [0., 0.], [0., -1.]], dtype="f")
        self.atol_threshold = 1e-4

    def test_relu_forward(self):
        relu = ReLU()
        A = relu.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_relu_backward(self):
        relu = ReLU()
        relu.forward(self.Z)
        dLdZ = relu.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


class TestGELU(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        self.dLdA = np.array([1.0, 1.0, 0, 1.0, -1.0])
        self.A_solution = np.array([-0.0455, -0.1543, 0.0, 0.3457, 1.9545])
        self.dLdZ_solution = np.array([-0.0852, 0.1325, 0, 0.8675, -1.0852])
        self.atol_threshold = 1e-4

    def test_gelu_forward(self):
        gelu = GELU()
        A = gelu.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_gelu_backward(self):
        gelu = GELU()
        gelu.forward(self.Z)
        dLdZ = gelu.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


class TestSoftmax(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[3.0, -4.5, 1.0, 6.5, 2.0], [-2.0, -0.5, 0.0, 0.5, 2.0]])
        self.dLdA = np.array([[2.0, -1.0, 3.0, -1.0, -2.0], [1.0, 1.0, 3.0, 1.0, -1.0]])
        self.A_solution = np.array([[0.0289, 0.0, 0.0039, 0.9566, 0.0106], [0.0126, 0.0563, 0.0928, 0.1529, 0.6855]])
        self.dLdZ_solution = np.array([[0.084, 0.0, 0.0153, -0.0877, -0.0116], [0.0149, 0.0667, 0.2955, 0.1813, -0.5584]])
        self.atol_threshold = 1e-4

    def test_softmax_forward(self):
        softmax = Softmax()
        A = softmax.forward(self.Z)
        np.testing.assert_allclose(A.round(4), self.A_solution, atol=self.atol_threshold)

    def test_softmax_backward(self):
        softmax = Softmax()
        softmax.forward(self.Z)
        dLdZ = softmax.backward(self.dLdA)
        np.testing.assert_allclose(dLdZ.round(4), self.dLdZ_solution, atol=self.atol_threshold)


if __name__ == '__main__':
    unittest.main()