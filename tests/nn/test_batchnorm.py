import unittest
import numpy as np
from mytorch.nn.batchnorm import BatchNorm1d

class TestBatchNorm1d(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.A = np.array([[1., 4.], [7., 0.], [1., 0.], [7., 4.]], dtype="f")
        self.BW = np.array([[2., 5.]], dtype="f")
        self.Bb = np.array([[-1., 2.]], dtype="f")
        self.dLdA = np.array([[-6., 2.], [-12., 16.], [-12., 20.], [-6., 2.]], dtype="f")

    def test_batchnorm_forward_eval(self):
        bn = BatchNorm1d(2)
        bn.BW = self.BW
        bn.Bb = self.Bb

        BZ = bn.forward(self.A, eval=True)
        BZ_solution = np.array([[1., 22.], [13., 2.], [1., 2.], [13., 22.]], dtype="f")
        self.assertTrue(np.allclose(BZ, BZ_solution, atol=self.atol_threshold))

    def test_batchnorm_forward_train(self):
        bn = BatchNorm1d(2)
        bn.BW = self.BW
        bn.Bb = self.Bb

        BZ = bn.forward(self.A, eval=False)
        BZ_solution = np.array([[-3., 7.], [1., -3.], [-3., -3.], [1., 7.]])
        self.assertTrue(np.allclose(BZ, BZ_solution, atol=self.atol_threshold))

    def test_batchnorm_backward(self):
        bn = BatchNorm1d(2)
        bn.BW = self.BW
        bn.Bb = self.Bb
        # Assuming the forward pass in training mode has been done
        bn.forward(self.A, eval=False)

        dLdZ = bn.backward(self.dLdA)
        dLdZ_solution = np.array([[2., 0.], [-2., -5.], [-2., 5.], [2., 0.]], dtype="f")
        self.assertTrue(np.allclose(dLdZ, dLdZ_solution, atol=self.atol_threshold))

if __name__ == '__main__':
    unittest.main()
