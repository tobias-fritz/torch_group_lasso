import unittest
import torch
from src.group_lasso import GroupLasso, apply_group_lasso, power_iteration

class TestGroupLassoFunctions(unittest.TestCase):
    def test_power_iteration(self):
        # Use a symmetric positive-definite matrix with known maximum eigenvalue.
        A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        max_eig = power_iteration(A)
        self.assertAlmostEqual(max_eig, 3.0, places=4)

    def test_apply_group_lasso(self):
        # Create a simple coefficients tensor and groups.
        coefficients = torch.tensor([3.0, 4.0, 0.5, 0.2])
        groups = torch.tensor([0, 0, 1, 1])
        # Apply function with chosen parameters.
        updated = apply_group_lasso(coefficients.clone(), groups, group_reg=1.0, step_size=0.1)
        # Verify coefficients are shrunken (non-increasing absolute values)
        self.assertTrue(torch.all(torch.abs(updated) <= torch.abs(coefficients) + 1e-4))

class TestGroupLassoModel(unittest.TestCase):
    def test_fit_predict(self):
        torch.manual_seed(42)
        n_samples = 100
        n_features = 10
        X = torch.randn(n_samples, n_features)
        true_coef = torch.randn(n_features)
        y = X @ true_coef + 0.1 * torch.randn(n_samples)
        groups = torch.tensor([i // 2 for i in range(n_features)], dtype=torch.int64)
        
        model = GroupLasso(feature_groups=groups,
                            group_penalty=0.1,
                            lasso_penalty=0.1,
                            scaling_penalty=0.1,
                            fit_intercept=True)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(predictions.shape[0], n_samples)
        
    def test_invalid_input(self):
        torch.manual_seed(0)
        n_samples = 50
        n_features = 8
        X = torch.randn(n_samples, n_features)
        y = torch.randn(n_samples + 1)
        groups = torch.arange(n_features, dtype=torch.int64)
        model = GroupLasso(feature_groups=groups)
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_cuda_device_unavailable(self):
        # Force a CUDA device when CUDA is not available.
        if torch.cuda.is_available():
            self.skipTest("CUDA is available, skipping device unavailability test")
        n_samples, n_features = 20, 4
        X = torch.randn(n_samples, n_features)
        y = torch.randn(n_samples)
        groups = torch.arange(n_features, dtype=torch.int64)
        with self.assertRaises(Exception):
            model = GroupLasso(feature_groups=groups, device='cuda')
            model.fit(X, y)

    def test_device_conversion(self):
        # Verify that model created with CUDA converts CPU tensors correctly.
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device conversion test")
        n_samples, n_features = 50, 6
        X = torch.randn(n_samples, n_features)  # CPU tensor
        y = torch.randn(n_samples)               # CPU tensor
        groups = torch.tensor([i % 3 for i in range(n_features)], dtype=torch.int64)
        model = GroupLasso(feature_groups=groups, fit_intercept=True, device='cuda')
        model.fit(X, y)
        predictions = model.predict(X)
        # Check that predictions are on CUDA device.
        self.assertEqual(predictions.device.type, 'cuda')

    def test_intercept_loss(self):
        # Create synthetic data with a known intercept.
        torch.manual_seed(123)
        n_samples, n_features = 100, 5
        X = torch.randn(n_samples, n_features)
        beta_true = torch.randn(n_features)
        true_intercept = 3.0
        y = X @ beta_true + true_intercept + 0.01 * torch.randn(n_samples)  # minimal noise
        groups = torch.arange(n_features, dtype=torch.int64)
        
        model = GroupLasso(feature_groups=groups, fit_intercept=True, max_iterations=500)
        model.fit(X, y)
        # Assert that the fitted intercept is close to the true intercept.
        self.assertAlmostEqual(model.intercept_.item(), true_intercept, places=1)

if __name__ == '__main__':
    unittest.main()
