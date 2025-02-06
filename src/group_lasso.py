# Implementing a group lasso algorithm with sparsity ussing pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp
import torch.jit
import time 

from typing import List, Tuple, Union

def power_iteration(A: torch.Tensor, num_iterations: int = 100, tol: float = 1e-6) -> float:
    """Compute largest eigenvalue using power iteration method."""
    n = A.shape[0]
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / torch.norm(v)
    
    for _ in range(num_iterations):
        v_new = A @ v
        norm = torch.norm(v_new)
        if norm < 1e-10:  # Handle near-zero case
            return 1.0
        v_new = v_new / norm
        if torch.norm(v_new - v) < tol:
            break
        v = v_new
    
    return (v @ (A @ v)).item()

@torch.jit.script
def apply_group_lasso(coefficients: torch.Tensor, 
                      groups: torch.Tensor, 
                      group_reg: float, 
                      step_size: float) -> torch.Tensor:
    # Assume groups is a 1D tensor of group indices and
    # coefficients is a 1D tensor; we loop over unique groups.
    unique_groups = torch.unique(groups)
    for i in range(unique_groups.size(0)):
        group_id = unique_groups[i]
        # Get indices where groups equal group_id
        mask = groups == group_id
        if mask.sum() > 0:
            group_coef = coefficients.masked_select(mask)
            group_norm = group_coef.norm()
            if group_norm > 1e-10:
                shrinkage_factor = torch.clamp(1 - group_reg * step_size / (group_norm + 1e-10), min=0.0)
                coefficients[mask] = group_coef * shrinkage_factor
    return coefficients

class GroupLasso(nn.Module):
    """Group Lasso with L1 regularization and adaptive scaling implementation using PyTorch.

    The optimization combines three regularization terms:
    1. Group Lasso penalty (λ₁): Encourages group sparsity
    2. L1 penalty (λ₂): Encourages individual feature sparsity
    3. Scaling penalty (λ₃): Controls the scale of predictions

    Parameters
    ----------
    feature_groups : torch.Tensor
        1D tensor of group indices for each feature
    group_penalty : float, default=0.1
        Group lasso penalty strength (λ₁)
    lasso_penalty : float, default=0.1
        L1 penalty strength (λ₂)
    scaling_penalty : float, default=0.1
        Scaling penalty strength (λ₃)
    subsampling_scheme : str, default='random'
        Method for subsampling during optimization
    fit_intercept : bool, default=False
        Whether to fit an intercept term
    random_seed : int, default=42
        Random seed for reproducibility
    warm_start : bool, default=False
        Whether to reuse previous solution
    max_iterations : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-4
        Convergence tolerance
    verbose : bool, default=False
        Whether to print convergence messages

    Attributes
    ----------
    coefficients_ : torch.Tensor
        Fitted coefficients (β)
    intercept_ : torch.Tensor
        Fitted intercept (γ)
    scaling_ : torch.Tensor
        Fitted scaling factor (s)

    Notes
    -----
    The optimization is solved using proximal gradient descent with the following steps:
    1. Gradient step on smooth part (least squares loss)
    2. Proximal operator for group lasso penalty
    3. Proximal operator for L1 penalty
    4. Update intercept and scaling

    Examples
    --------
    >>> import torch
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> X = torch.randn(100, 10).to(device)
    >>> y = torch.randn(100).to(device)
    >>> groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    >>> model = GroupLasso(feature_groups=groups, group_penalty=0.1, lasso_penalty=0.1, scaling_penalty=0.1, fit_intercept=True, device=device)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """
    
    def __init__(self, 
                 feature_groups: torch.Tensor,
                 group_penalty: float = 0.1,
                 lasso_penalty: float = 0.1,
                 scaling_penalty: float = 0.1,
                 subsampling_scheme: str = 'random',
                 fit_intercept: bool = False,
                 random_seed: int = 42,
                 warm_start: bool = False,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-4,
                 verbose: bool = False,
                 device: str = None
                 ) -> None:
        
        # Validate inputs
        if not isinstance(feature_groups, torch.Tensor):
            raise TypeError("feature_groups must be a torch.Tensor")
        if not feature_groups.dtype in [torch.int32, torch.int64]:
            raise TypeError("feature_groups must contain integer values")
        if feature_groups.dim() != 1:
            raise ValueError("feature_groups must be 1-dimensional")
            
        for param_name, param_value in [
            ('group_penalty', group_penalty),
            ('lasso_penalty', lasso_penalty),
            ('scaling_penalty', scaling_penalty),
            ('tolerance', tolerance)
        ]:
            if not isinstance(param_value, (int, float)):
                raise TypeError(f"{param_name} must be a number")
            if param_value < 0:
                raise ValueError(f"{param_name} must be non-negative")
                
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")

        super(GroupLasso, self).__init__()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.groups = feature_groups.to(self.device)
        self.group_reg = group_penalty
        self.l1_reg = lasso_penalty
        self.scale_reg = scaling_penalty
        self.subsampling_scheme = subsampling_scheme
        self.fit_intercept = fit_intercept
        self.random_state = random_seed
        self.warm_start = warm_start
        self.max_iter = max_iterations
        self.tol = tolerance
        self.verbose = verbose

        self.coefficients_ = None
        self.intercept_ = None
        self.scaling_ = None

    def _update_progress(self, iteration: int, total_iterations: int, current_iter_time: float, iter_times: list) -> None:
        # Calculate average over the last up-to 10 iterations
        last_times = iter_times[-10:] if len(iter_times) >= 10 else iter_times
        avg_time = sum(last_times) / len(last_times) if last_times else 0
        remaining = total_iterations - iteration - 1
        est_remaining = remaining * avg_time
        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{total_iterations} - Iteration time: {current_iter_time:.4f} sec, "
                f"Estimated remaining time: {est_remaining:.2f} sec")
    
    def fit(self, features: torch.Tensor, target: torch.Tensor) -> 'GroupLasso':
        # Input validation
        if not isinstance(features, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise TypeError("features and target must be torch.Tensor")
            
        if features.dim() != 2:
            raise ValueError("features must be 2-dimensional")
        if target.dim() != 1:
            raise ValueError("target must be 1-dimensional")
            
        n_samples, n_features = features.shape
        if n_samples != target.shape[0]:
            raise ValueError(f"features and target have incompatible shapes: {features.shape} and {target.shape}")
        if n_features != len(self.groups):
            raise ValueError(f"Number of features {n_features} does not match length of groups {len(self.groups)}")
            
        if torch.isnan(features).any() or torch.isnan(target).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(features).any() or torch.isinf(target).any():
            raise ValueError("Input contains infinite values")

        # Move data to device
        features = features.to(self.device)
        target = target.to(self.device)
        
        # Normalize features with improved stability
        feature_norm = torch.sqrt(torch.sum(features ** 2, dim=0, keepdim=True) + 1e-8)
        feature_norm = torch.maximum(feature_norm, torch.ones_like(feature_norm, device=self.device) * 1e-8)
        normalized_features = features / feature_norm

        # Initialize parameters if not warm started
        if not self.warm_start or self.coefficients_ is None:
            self.coefficients_ = torch.zeros(n_features, requires_grad=True, device=self.device)
            self.scaling_ = torch.ones(1, requires_grad=True, device=self.device)
            self.intercept_ = torch.zeros(1, requires_grad=False, device=self.device)

        try:
            # Compute step size using multiple methods with fallback
            try:
                # Try power iteration first
                feature_gram = normalized_features.T @ normalized_features / n_samples
                max_eigval = power_iteration(feature_gram)
                if not (0 < max_eigval < float('inf')):
                    raise ValueError("Invalid eigenvalue")
                step_size = 1.0 / (max_eigval + 1e-8)
            except Exception:
                # Fallback to simpler Lipschitz estimate
                norm_X = torch.norm(normalized_features, p=2)
                if norm_X > 0:
                    step_size = 1.0 / (norm_X ** 2 / n_samples + 1e-8)
                else:
                    step_size = 0.1  # Conservative default

            # Ensure step size is within reasonable bounds
            step_size = min(max(step_size, 1e-10), 1.0)

            # Use torch.amp instead of torch.cuda.amp to avoid deprecation warnings
            use_amp = self.device.type == 'cuda'
            scaler = torch.amp.GradScaler(enabled=use_amp)  # updated GradScaler usage

            iter_times = []  # list to track iteration durations
            previous_coefficients = torch.full_like(self.coefficients_, float('inf'), device=self.device)
            
            for iteration in range(self.max_iter):
                iter_start = time.time()
                if use_amp:
                    # Updated autocast usage with required device_type argument.
                    with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                        predictions = normalized_features @ self.coefficients_
                        predictions = predictions + self.intercept_
                        predictions *= torch.clamp(self.scaling_, min=1e-8, max=1e8)
                        residuals = target - predictions
                        gradient = -2 * self.scaling_ * normalized_features.T @ residuals / n_samples
                        gradient = torch.clamp(gradient, min=-1e8, max=1e8)
                else:
                    # ...existing code...
                    predictions = normalized_features @ self.coefficients_
                    predictions = predictions + self.intercept_
                    predictions *= torch.clamp(self.scaling_, min=1e-8, max=1e8)
                    residuals = target - predictions
                    gradient = -2 * self.scaling_ * normalized_features.T @ residuals / n_samples
                    gradient = torch.clamp(gradient, min=-1e8, max=1e8)
                
                # NEW: Sanitize the gradient to avoid NaN or infinite values.
                gradient = torch.nan_to_num(gradient, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Gradient descent step
                self.coefficients_ = self.coefficients_ - step_size * gradient

                # NEW: Sanitize coefficients after update.
                self.coefficients_ = torch.nan_to_num(self.coefficients_, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Replace Python for-loop with TorchScript function
                self.coefficients_ = apply_group_lasso(self.coefficients_, self.groups, self.group_reg, step_size)

                # Apply L1 proximal operator
                self.coefficients_ = torch.sign(self.coefficients_) * torch.maximum(
                    torch.abs(self.coefficients_) - self.l1_reg * step_size,
                    torch.tensor(0.0, device=self.device)
                )

                # NEW: Clip coefficients to prevent divergence and avoid infinite values.
                self.coefficients_ = torch.clamp(self.coefficients_, min=-1e10, max=1e10)

                # Update intercept if needed
                if self.fit_intercept:
                    self.intercept_ = torch.mean(target - normalized_features @ self.coefficients_)

                # Update scale with bounds
                if self.scale_reg > 0:
                    residuals = target - (normalized_features @ self.coefficients_ + self.intercept_)
                    scale = torch.mean(residuals ** 2) / (2 * self.scale_reg)
                    self.scaling_ = torch.clamp(scale, min=1e-8, max=1e8)

                # Check for numerical instability
                if torch.isnan(self.coefficients_).any():
                    raise ValueError("NaN values in coefficients")
                if torch.isinf(self.coefficients_).any():
                    raise ValueError("Infinite values in coefficients")

                # Check convergence with safe norm computation
                coef_diff = torch.norm(self.coefficients_ - previous_coefficients)
                prev_norm = torch.norm(previous_coefficients)
                coefficient_change = coef_diff / (prev_norm + 1e-10)
                
                iter_elapsed = time.time() - iter_start  # measure iteration duration
                iter_times.append(iter_elapsed)
                if self.verbose:
                    self._update_progress(iteration, self.max_iter, iter_elapsed, iter_times)
                
                if coefficient_change < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration}")
                    break

                previous_coefficients = self.coefficients_.clone()

            # Denormalize coefficients
            self.coefficients_ = self.coefficients_ / feature_norm.squeeze()

        except Exception as error:
            self.coefficients_ = None
            self.intercept_ = torch.zeros(1, requires_grad=False, device=self.device)
            self.scaling_ = torch.ones(1, requires_grad=False, device=self.device)
            raise RuntimeError(f"Optimization failed: {str(error)}")

        return self

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        # Check if model is fitted
        if self.coefficients_ is None:
            raise RuntimeError("Model must be fitted before prediction")
            
        # Input validation
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be torch.Tensor")
        if features.dim() != 2:
            raise ValueError("features must be 2-dimensional")
        if features.shape[1] != len(self.groups):
            raise ValueError(f"features has {features.shape[1]} features but model was trained with {len(self.groups)} features")
            
        # Move features to device
        features = features.to(self.device)
        
        predictions = features @ self.coefficients_
        predictions = predictions + self.intercept_  # Now safe since intercept is always initialized
        predictions *= self.scaling_
        return predictions

    def score(self, features: torch.Tensor, target: torch.Tensor) -> float:
        """Compute mean squared error (MSE) of predictions."""
        # Input validation
        if not isinstance(target, torch.Tensor):
            raise TypeError("target must be torch.Tensor")
        if target.dim() != 1:
            raise ValueError("target must be 1-dimensional")
        if features.shape[0] != target.shape[0]:
            raise ValueError("features and target have incompatible shapes")
            
        predictions = self.predict(features)
        return torch.mean((target - predictions) ** 2).item()
