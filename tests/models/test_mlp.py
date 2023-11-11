import unittest
import pickle
import os
import numpy as np
from mytorch.models.mlp import MLP0, MLP1, MLP4

pkl_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_data')

class TestMLP0(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.A0 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.]], dtype="f")
        self.W0 = np.array([
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.]], dtype="f")
        self.b0 = np.array([
            [-1.],
            [ 0.],
            [ 1.]], dtype="f")
        self.dLdA1 = np.array([
            [-4., -3., -2.],
            [-1., -0.,  1.],
            [ 2.,  3.,  4.],
            [ 5.,  6.,  7.]], dtype="f")

    def test_mlp0_forward_backward(self):
        mlp0 = MLP0(debug=True)
        mlp0.layers[0].W = self.W0
        mlp0.layers[0].b = self.b0

        # Forward pass
        A1 = mlp0.forward(self.A0)
        Z0 = mlp0.Z0

        # Expected forward pass outputs
        Z0_solution = np.array([
            [ 10.,  -3., -16.],
            [  4.,  -1.,  -6.],
            [ -2.,   1.,   4.],
            [ -8.,   3.,  14.]], dtype="f")
        A1_solution =np.array([
            [10.,  0.,  0.],
            [ 4.,  0.,  0.],
            [ 0.,  1.,  4.],
            [ 0.,  3., 14.]], dtype="f")

        # Backward pass
        mlp0.backward(self.dLdA1)
        dLdZ0 = mlp0.dLdZ0
        dLdA0 = mlp0.dLdA0
        dLdW0 = mlp0.layers[0].dLdW
        dLdb0 = mlp0.layers[0].dLdb

        # Expected backward pass outputs
        dLdZ0_solution = np.array([
            [-4., -0., -0.],
            [-1., -0.,  0.],
            [ 0.,  3.,  4.],
            [ 0.,  6.,  7.]], dtype="f")
        dLdA0_solution = np.array([
            [ 8.,  4.],
            [ 2.,  1.],
            [ 8., 15.],
            [14., 27.]], dtype="f")
        dLdW0_solution = np.array([
            [4.5,  3.25],
            [3. ,  5.25],
            [3.5,  6.25]], dtype="f")
        dLdb0_solution = np.array([
            [-1.25],
            [ 2.25],
            [ 2.75]], dtype="f")

        # Assertions
        self.assertTrue(np.allclose(Z0, Z0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(A1, A1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdZ0, dLdZ0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdA0, dLdA0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdW0, dLdW0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdb0, dLdb0_solution, atol=self.atol_threshold))


class TestMLP1(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.A0 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.]], dtype="f")
        self.W0 = np.array([
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.]], dtype="f")
        self.b0 = np.array([
            [-1.],
            [ 0.],
            [ 1.]], dtype="f")
        self.W1 = np.array([
            [-2., -1., 0],
            [ 1.,  2., 3]], dtype="f")
        self.b1 = np.array([
            [-1.],
            [ 1.]], dtype="f")
        self.dLdA2 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.]], dtype="f")
        
    def test_mlp1_forward_backward(self):
        mlp1 = MLP1(debug=True)
        mlp1.layers[0].W = self.W0
        mlp1.layers[0].b = self.b0
        mlp1.layers[1].W = self.W1
        mlp1.layers[1].b = self.b1

        # Forward pass
        A2 = mlp1.forward(self.A0)
        Z0 = mlp1.Z0
        A1 = mlp1.A1
        Z1 = mlp1.Z1

        # Expected forward pass outputs
        Z0_solution = np.array([
            [ 10.,  -3., -16.],
            [  4.,  -1.,  -6.],
            [ -2.,   1.,   4.],
            [ -8.,   3.,  14.]], dtype="f")
        A1_solution = np.array([
            [10.,  0.,  0.],
            [ 4.,  0.,  0.],
            [ 0.,  1.,  4.],
            [ 0.,  3., 14.]], dtype="f")
        Z1_solution = np.array([
            [-21.,  11.],
            [ -9.,   5.],
            [ -2.,  15.],
            [ -4.,  49.]], dtype="f")
        A2_solution = np.array([
            [ 0., 11.],
            [ 0.,  5.],
            [ 0., 15.],
            [ 0., 49.]], dtype="f")
        
        # Backward pass
        mlp1.backward(self.dLdA2)
        dLdZ1 = mlp1.dLdZ1
        dLdA1 = mlp1.dLdA1
        dLdZ0 = mlp1.dLdZ0
        dLdA0 = mlp1.dLdA0
        dLdW0 = mlp1.layers[0].dLdW
        dLdb0 = mlp1.layers[0].dLdb

        # Expected backward pass outputs
        dLdZ1_solution = np.array([
            [-0., -3.],
            [-0., -1.],
            [ 0.,  1.],
            [ 0.,  3.]], dtype="f")
        dLdA1_solution = np.array([
            [-3., -6., -9.],
            [-1., -2., -3.],
            [ 1.,  2.,  3.],
            [ 3.,  6.,  9.]], dtype="f")
        dLdZ0_solution = np.array([
            [-3., -0., -0.],
            [-1., -0., -0.],
            [ 0.,  2.,  3.],
            [ 0.,  6.,  9.]], dtype="f")
        dLdA0_solution = np.array([
            [ 6.,  3.],
            [ 2.,  1.],
            [ 6., 11.],
            [18., 33.]], dtype="f")
        dLdW0_solution = np.array([
            [3.5, 2.5],
            [3. , 5. ],
            [4.5, 7.5]], dtype="f")
        dLdb0_solution = np.array([
            [-1.],
            [ 2.],
            [ 3.]], dtype="f")
        
        # Assertions
        self.assertTrue(np.allclose(Z0, Z0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(A1, A1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(Z1, Z1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(A2, A2_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdZ1, dLdZ1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdA1, dLdA1_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdZ0, dLdZ0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdA0, dLdA0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdW0, dLdW0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdb0, dLdb0_solution, atol=self.atol_threshold))


class TestMLP4(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.A0 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]], dtype="f")
        with open(f"{pkl_dir}/mlp4_W.pkl", "rb") as f:
            self.W0, self.W1, self.W2, self.W3, self.W4 = pickle.load(f)
        with open(f"{pkl_dir}/mlp4_b.pkl", "rb") as f:
            self.b0, self.b1, self.b2, self.b3, self.b4 = pickle.load(f)
        self.dLdA5 = np.array([
            [-4., -3.],
            [-2., -1.],
            [ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]], dtype="f")

    def test_mlp4_forward_backward(self):
        mlp4 = MLP4(debug=True)
        for i in range(len(mlp4.layers)):
            mlp4.layers[i].W = getattr(self, f"W{i}")
            mlp4.layers[i].b = getattr(self, f"b{i}")
        
        # Forward pass
        A5 = mlp4.forward(self.A0)

        # Expected forward pass outputs
        with open(f"{pkl_dir}/mlp4_sol_Z.pkl", "rb") as f:
            mlp4_sol_Z = pickle.load(f)
        with open(f"{pkl_dir}/mlp4_sol_A.pkl", "rb") as f:
            mlp4_sol_A = pickle.load(f)

        # Backward pass
        mlp4.backward(self.dLdA5)
        dLdW0, dLdb0 = mlp4.layers[0].dLdW, mlp4.layers[0].dLdb

        # Expected backward pass outputs
        with open(f"{pkl_dir}/mlp4_sol_dLdZ.pkl", "rb") as f:
            mlp4_sol_dLdZ = pickle.load(f)
        with open(f"{pkl_dir}/mlp4_sol_dLdA.pkl", "rb") as f:
            mlp4_sol_dLdA = pickle.load(f)
        with open(f"{pkl_dir}/mlp4_sol_dLdW.pkl", "rb") as f:
            dLdW0_solution = pickle.load(f)[0]
        with open(f"{pkl_dir}/mlp4_sol_dLdb.pkl", "rb") as f:
            dLdb0_solution = pickle.load(f)[0]
        
        # Assertions
        for get_Z, sol_Z in zip(mlp4.Z, mlp4_sol_Z):
            self.assertTrue(np.allclose(get_Z, sol_Z, atol=self.atol_threshold))
        for get_A, sol_A in zip(mlp4.A[1:], mlp4_sol_A):
            self.assertTrue(np.allclose(get_A, sol_A, atol=self.atol_threshold))
        for get_dLdZ, sol_dLdZ in zip(mlp4.dLdZ, mlp4_sol_dLdZ):
            self.assertTrue(np.allclose(get_dLdZ, sol_dLdZ, atol=self.atol_threshold))
        for get_dLdA, sol_dLdA in zip(mlp4.dLdA, mlp4_sol_dLdA):
            self.assertTrue(np.allclose(get_dLdA, sol_dLdA, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdW0, dLdW0_solution, atol=self.atol_threshold))
        self.assertTrue(np.allclose(dLdb0, dLdb0_solution, atol=self.atol_threshold))
        

if __name__ == '__main__':
    unittest.main()
