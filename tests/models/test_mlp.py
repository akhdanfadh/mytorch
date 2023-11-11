import unittest
import numpy as np
import torch
from mytorch.models.mlp import MLP0, MLP1, MLP4

class TestMLP0(unittest.TestCase):
    
    def setUp(self):
        self.atol_threshold = 1e-4
        self.n_tests = 5  # Number of test cases

    def test_mlp0_forward_backward(self):
        for i in range(self.n_tests):
            A0 = np.random.randn(4, 2).astype("f")
            W0 = np.random.randn(3, 2).astype("f")
            b0 = np.random.randn(3, ).astype("f")
            A0_tensor = torch.tensor(A0, requires_grad=True)

            # Torch linear for correct answer
            torch_linear = torch.nn.Linear(2, 3)
            torch_linear.weight.data = torch.tensor(W0)
            torch_linear.bias.data = torch.tensor(b0)
            torch_linear.requires_grad_()
            Z0_tensor = torch_linear(A0_tensor)
            A1_tensor = torch.relu(Z0_tensor)

            # Student's MLP0
            mlp0 = MLP0(debug=True)
            mlp0.layers[0].W = W0
            mlp0.layers[0].b = b0.reshape(-1, 1)

            A1_ = mlp0.forward(A0)
            Z0_ = mlp0.Z0

            # Check forward pass
            self.assertTrue(np.allclose(Z0_tensor.detach().numpy(), Z0_, atol=self.atol_threshold))
            self.assertTrue(np.allclose(A1_tensor.detach().numpy(), A1_, atol=self.atol_threshold))

            # Backward pass
            dLdA1 = np.random.randn(4, 3).astype("f")
            dA1dZ0 = torch.autograd.grad(A1_tensor, Z0_tensor, grad_outputs=torch.ones_like(A1_tensor))[0].numpy()
            dLdZ0 = dLdA1 * dA1dZ0
            Z0_tensor.backward(gradient=torch.tensor(dLdZ0))

            mlp0.backward(dLdA1)

            # Check backward pass
            self.assertTrue(np.allclose(dLdZ0, mlp0.dLdZ0, atol=self.atol_threshold))
            self.assertTrue(np.allclose(A0_tensor.grad.data.numpy(), mlp0.dLdA0, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear.weight.grad.data.numpy(), mlp0.layers[0].dLdW, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear.bias.grad.data.numpy().reshape(-1, 1), mlp0.layers[0].dLdb, atol=self.atol_threshold))


class TestMLP1(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.n_tests = 5  # Number of test cases

    def test_mlp1_forward_backward(self):
        for i in range(self.n_tests):
            A0 = np.random.randn(4, 2).astype("f")
            W0 = np.random.randn(3, 2).astype("f")
            b0 = np.random.randn(3, ).astype("f")
            W1 = np.random.randn(2, 3).astype("f")
            b1 = np.random.randn(2, ).astype("f")
            A0_tensor = torch.tensor(A0, requires_grad=True)

            # Torch linear for correct answer
            torch_linear0 = torch.nn.Linear(2, 3)
            torch_linear0.weight.data = torch.tensor(W0)
            torch_linear0.bias.data = torch.tensor(b0)
            torch_linear0.requires_grad_()
            torch_linear1 = torch.nn.Linear(3, 2)
            torch_linear1.weight.data = torch.tensor(W1)
            torch_linear1.bias.data = torch.tensor(b1)
            torch_linear1.requires_grad_()
            Z0_tensor = torch_linear0(A0_tensor)
            A1_tensor = torch.relu(Z0_tensor)
            A1_tensor_copy = torch.tensor(A1_tensor.detach().numpy(), requires_grad=True)
            Z1_tensor = torch_linear1(A1_tensor_copy)
            A2_tensor = torch.relu(Z1_tensor)

            # Student's MLP1
            mlp1 = MLP1(debug=True)
            mlp1.layers[0].W = W0
            mlp1.layers[0].b = b0.reshape(-1, 1)
            mlp1.layers[2].W = W1
            mlp1.layers[2].b = b1.reshape(-1, 1)

            A2_ = mlp1.forward(A0)

            # Check forward pass
            self.assertTrue(np.allclose(Z0_tensor.detach().numpy(), mlp1.Z0, atol=self.atol_threshold))
            self.assertTrue(np.allclose(A1_tensor.detach().numpy(), mlp1.A1, atol=self.atol_threshold))
            self.assertTrue(np.allclose(Z1_tensor.detach().numpy(), mlp1.Z1, atol=self.atol_threshold))
            self.assertTrue(np.allclose(A2_tensor.detach().numpy(), A2_, atol=self.atol_threshold))

            # Backward pass
            dLdA2 = np.random.randn(4, 2).astype("f")
            dA2dZ1 = torch.autograd.grad(A2_tensor, Z1_tensor, grad_outputs=torch.ones_like(A2_tensor))[0].numpy()
            dLdZ1 = dLdA2 * dA2dZ1
            Z1_tensor.backward(gradient=torch.tensor(dLdZ1), retain_graph=True)
            dLdA1 = A1_tensor_copy.grad.data.numpy()
            dA1dZ0 = torch.autograd.grad(A1_tensor, Z0_tensor, grad_outputs=torch.ones_like(A1_tensor))[0].numpy()
            dLdZ0 = dLdA1 * dA1dZ0
            Z0_tensor.backward(gradient=torch.tensor(dLdZ0), retain_graph=True)
            dLdA0 = A0_tensor.grad.data.numpy()

            mlp1.backward(dLdA2)

            # Check backward pass
            self.assertTrue(np.allclose(dLdZ1, mlp1.dLdZ1, atol=self.atol_threshold))
            self.assertTrue(np.allclose(dLdA1, mlp1.dLdA1, atol=self.atol_threshold))
            self.assertTrue(np.allclose(dLdZ0, mlp1.dLdZ0, atol=self.atol_threshold))
            self.assertTrue(np.allclose(dLdA0, mlp1.dLdA0, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear0.weight.grad.data.numpy(), mlp1.layers[0].dLdW, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear0.bias.grad.data.numpy().reshape(-1, 1), mlp1.layers[0].dLdb, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear1.weight.grad.data.numpy(), mlp1.layers[2].dLdW, atol=self.atol_threshold))
            self.assertTrue(np.allclose(torch_linear1.bias.grad.data.numpy().reshape(-1, 1), mlp1.layers[2].dLdb, atol=self.atol_threshold))


class TestMLP4(unittest.TestCase):

    def setUp(self):
        self.atol_threshold = 1e-4
        self.n_tests = 5  # Number of test cases

    def test_mlp4_forward_backward(self):
        for i in range(self.n_tests):
            A0 = np.random.randn(4, 2).astype("f")
            W0, W1, W2, W3, W4 = [np.random.randn(*shape).astype("f") for shape in [(4, 2), (8, 4), (8, 8), (4, 8), (2, 4)]]
            b0, b1, b2, b3, b4 = [np.random.randn(size).astype("f") for size in [4, 8, 8, 4, 2]]
            A0_tensor = torch.tensor(A0, requires_grad=True)

            # Setting up torch layers for correct answers
            torch_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in [(2, 4), (4, 8), (8, 8), (8, 4), (4, 2)]]
            for layer, W, b in zip(torch_layers, [W0, W1, W2, W3, W4], [b0, b1, b2, b3, b4]):
                layer.weight.data = torch.tensor(W)
                layer.bias.data = torch.tensor(b)
                layer.requires_grad_()

            # Forward pass
            A_tensor = A0_tensor
            for layer in torch_layers:
                Z_tensor = layer(A_tensor)
                A_tensor = torch.relu(Z_tensor)

            # Student's MLP4
            mlp4 = MLP4(debug=True)
            mlp4.layers[0].W, mlp4.layers[2].W, mlp4.layers[4].W, mlp4.layers[6].W, mlp4.layers[8].W = W0, W1, W2, W3, W4
            mlp4.layers[0].b, mlp4.layers[2].b, mlp4.layers[4].b, mlp4.layers[6].b, mlp4.layers[8].b = b0.reshape(-1, 1), b1.reshape(-1, 1), b2.reshape(-1, 1), b3.reshape(-1, 1), b4.reshape(-1, 1)
            A5_ = mlp4.forward(A0)

            # Check forward pass
            self.assertTrue(np.allclose(A_tensor.detach().numpy(), A5_, atol=self.atol_threshold))

            # Backward pass
            dLdA5 = np.random.randn(4, 2).astype("f")
            dLdA5_tensor = torch.tensor(dLdA5)
            A_tensor.backward(dLdA5_tensor)
            mlp4.backward(dLdA5)

            # Check backward pass
            for layer, mlp_layer in zip(torch_layers, mlp4.layers[::2]):  # Compare only the linear layers
                self.assertTrue(np.allclose(layer.weight.grad.data.numpy(), mlp_layer.dLdW, atol=self.atol_threshold))
                self.assertTrue(np.allclose(layer.bias.grad.data.numpy().reshape(-1, 1), mlp_layer.dLdb, atol=self.atol_threshold))


if __name__ == '__main__':
    unittest.main()
