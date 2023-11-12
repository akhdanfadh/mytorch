import unittest
import os
import pickle
import numpy as np
from mytorch.nn.dropout import Dropout

pkl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')

def reset_prng():
    np.random.seed(11785)

class TestDropout(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.p = 0.3

        reset_prng()
        self.input = np.random.randn(20, 64)
        self.delta = self.input.copy()

    def test_droupout_forward(self):
        # Student outputs
        reset_prng()
        dropout = Dropout(p=self.p)
        output = dropout.forward(self.input)

        # Expected outputs
        with open(f"{pkl_dir}/dropout_sol_forward.pkl", "rb") as f:
            target = pickle.load(f)

        # Assertions
        self.assertTrue(np.allclose(output, target, atol=self.atol_threshold))
    
    def test_dropout_backward(self):
        # Student outputs
        reset_prng()
        dropout = Dropout(p=self.p)
        dropout.forward(self.input)
        output = dropout.backward(self.delta)

        # Expected outputs
        with open(f"{pkl_dir}/dropout_sol_backward.pkl", "rb") as f:
            target = pickle.load(f)

        # Assertions
        self.assertTrue(np.allclose(output, target, atol=self.atol_threshold))


if __name__ == '__main__':
    unittest.main()