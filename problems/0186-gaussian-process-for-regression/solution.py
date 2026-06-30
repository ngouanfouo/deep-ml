import math
import numpy as np

def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    """Matern kernel function."""
    # Compute Euclidean distance for multi-dimensional inputs
    r = np.linalg.norm(np.array(x) - np.array(x_prime))
    if nu == 0.5:
        return np.exp(-r / length_scale)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * r / length_scale) * np.exp(-np.sqrt(3) * r / length_scale)
    elif nu == 2.5:
        return (1 + np.sqrt(5) * r / length_scale + 5 * r**2 / (3 * length_scale**2)) * np.exp(-np.sqrt(5) * r / length_scale)
    else:
        raise ValueError("nu must be 0.5, 1.5, or 2.5")


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    """Radial Basis Function (RBF) kernel."""
    # Compute Euclidean distance for multi-dimensional inputs
    r = np.linalg.norm(np.array(x) - np.array(x_prime))
    return sigma**2 * np.exp(-r**2 / (2 * length_scale**2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    """Periodic kernel."""
    # For multi-dimensional, use Euclidean distance
    r = np.linalg.norm(np.array(x) - np.array(x_prime))
    return sigma**2 * np.exp(-2 * np.sin(np.pi * r / period)**2 / length_scale**2)


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    """Linear kernel."""
    x = np.array(x)
    x_prime = np.array(x_prime)
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    """Rational Quadratic kernel."""
    r = np.linalg.norm(np.array(x) - np.array(x_prime))
    return sigma**2 * (1 + r**2 / (2 * alpha * length_scale**2))**(-alpha)


# --- BASE CLASS -------------------------------------------------------------
class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        """
        Initialize Gaussian Process base class.
        
        Args:
            kernel: String specifying kernel type ('rbf', 'matern', 'periodic', 'linear', 'rational_quadratic')
            noise: Noise variance (added to diagonal of covariance matrix)
            kernel_params: Dictionary of kernel parameters
        """
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None
        self.L = None
    
    def _select_kernel(self, x1, x2):
        """Selects and computes the kernel value for two single data points."""
        if self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == 'matern':
            return matern_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == 'periodic':
            return periodic_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == 'linear':
            return linear_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == 'rational_quadratic':
            return rational_quadratic_kernel(x1, x2, **self.kernel_params)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def _compute_covariance(self, X1, X2):
        """
        Computes the covariance matrix between two sets of points.
        Supports multi-dimensional inputs.
        """
        # Ensure inputs are 2D arrays
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        
        # For RBF kernel - vectorized with proper broadcasting
        if self.kernel_name == 'rbf':
            sigma = self.kernel_params.get('sigma', 1.0)
            length_scale = self.kernel_params.get('length_scale', 1.0)
            
            # Compute pairwise squared distances
            # X1: (n1, d), X2: (n2, d)
            # We want: (n1, n2) matrix of squared distances
            # Use: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            
            # Compute squared norms
            X1_sq = np.sum(X1**2, axis=1, keepdims=True)  # (n1, 1)
            X2_sq = np.sum(X2**2, axis=1, keepdims=True)  # (n2, 1)
            
            # Compute pairwise squared distances
            # X1_sq: (n1, 1), X2_sq.T: (1, n2), X1 @ X2.T: (n1, n2)
            sq_dist = X1_sq + X2_sq.T - 2 * X1 @ X2.T
            
            # Ensure non-negative (numerical stability)
            sq_dist = np.maximum(sq_dist, 0)
            
            K = sigma**2 * np.exp(-sq_dist / (2 * length_scale**2))
            return K
        
        # For linear kernel - vectorized
        elif self.kernel_name == 'linear':
            sigma_b = self.kernel_params.get('sigma_b', 1.0)
            sigma_v = self.kernel_params.get('sigma_v', 1.0)
            # X1: (n1, d), X2: (n2, d) -> K: (n1, n2)
            K = sigma_b**2 + sigma_v**2 * X1 @ X2.T
            return K
        
        # For Matern kernel - vectorized with element-wise computation
        elif self.kernel_name == 'matern':
            length_scale = self.kernel_params.get('length_scale', 1.0)
            nu = self.kernel_params.get('nu', 1.5)
            K = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = matern_kernel(X1[i], X2[j], length_scale, nu)
            return K
        
        # For other kernels - element-wise computation
        else:
            K = np.zeros((n1, n2))
            for i in range(n1):
                for j in range(n2):
                    K[i, j] = self._select_kernel(X1[i], X2[j])
            return K


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        """
        Fit the Gaussian Process regression model.
        
        Args:
            X: Training input data (n_samples, n_features)
            y: Training target values (n_samples,)
        """
        # Ensure X is 2D
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train = X
        
        # Ensure y is 1D
        y = np.array(y)
        if y.ndim > 1:
            y = y.ravel()
        self.y_train = y
        
        # Compute covariance matrix
        K = self._compute_covariance(self.X_train, self.X_train)
        
        # Add noise to diagonal for numerical stability
        K += self.noise * np.eye(K.shape[0])
        
        # Compute Cholesky decomposition for numerical stability
        try:
            self.L = np.linalg.cholesky(K)
            # Solve L @ L.T @ alpha = y
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
            self.K_inv = None
        except np.linalg.LinAlgError:
            # Fallback to direct inversion if Cholesky fails
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ self.y_train
            self.L = None

    def predict(self, X_test, return_std=False):
        """
        Predict output values for test points.
        
        Args:
            X_test: Test input data (n_samples, n_features)
            return_std: If True, return standard deviation as well
        
        Returns:
            mu: Predicted means
            std: Predicted standard deviations (if return_std=True)
        """
        # Ensure X_test is 2D
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Compute covariance between test and training points
        K_s = self._compute_covariance(X_test, self.X_train)
        
        # Compute covariance between test points
        K_ss = self._compute_covariance(X_test, X_test)
        
        # Predict mean
        mu = K_s @ self.alpha
        
        # Predict variance
        if self.L is not None:
            # Using Cholesky decomposition
            v = np.linalg.solve(self.L, K_s.T)
            var = K_ss - v.T @ v
        else:
            # Using direct inversion
            var = K_ss - K_s @ self.K_inv @ K_s.T
        
        # Add noise to diagonal
        var += self.noise * np.eye(var.shape[0])
        
        # Ensure variance is non-negative (numerical stability)
        var = np.maximum(var, 0)
        
        mu = mu.flatten()
        
        if return_std:
            return mu, np.sqrt(np.diag(var))
        return mu

    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood of the training data.
        
        Returns:
            log_likelihood: Log marginal likelihood value
        """
        if self.L is not None:
            # Using Cholesky decomposition
            log_det = 2 * np.sum(np.log(np.diag(self.L)))
            n = len(self.y_train)
            log_likelihood = -0.5 * self.y_train @ self.alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        else:
            # Using direct inversion
            K = self._compute_covariance(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
            log_det = np.linalg.slogdet(K)[1]
            n = len(self.y_train)
            log_likelihood = -0.5 * self.y_train @ self.K_inv @ self.y_train - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        
        return float(log_likelihood)

    def optimize_hyperparameters(self, bounds=None, method='L-BFGS-B', max_iter=100):
        """
        Optimize hyperparameters by maximizing the log marginal likelihood.
        
        Args:
            bounds: Dictionary with parameter bounds {'param_name': (min, max)}
            method: Optimization method
            max_iter: Maximum number of iterations
        
        Returns:
            optimized_params: Dictio