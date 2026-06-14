import numpy as np
import torch
import torch.nn as nn

class FeatureDeconfounder(nn.Module):
    """
    Removes the linear influence of metadata/confounding variables from features.
    """
    
    def __init__(self):
        super().__init__()
        self.Sigma_inv = None
        self.is_fitted = False
    
    def fit(self, metadata):
        """
        Precompute inverse covariance from training metadata.
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(metadata, np.ndarray):
            metadata = torch.from_numpy(metadata).float()
        elif not isinstance(metadata, torch.Tensor):
            metadata = torch.tensor(metadata, dtype=torch.float32)
        
        # Compute Sigma = X^T @ X
        Sigma = metadata.T @ metadata
        
        # Add regularization for numerical stability
        reg = 1e-5
        K = Sigma.shape[0]
        Sigma_reg = Sigma + reg * torch.eye(K, device=Sigma.device)
        
        # Compute and store inverse
        self.Sigma_inv = torch.linalg.inv(Sigma_reg)
        self.is_fitted = True
    
    def transform(self, features, metadata):
        """
        Remove metadata influence from features.
        """
        assert self.is_fitted, "Must call fit() before transform()"
        
        # Convert numpy arrays to torch tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(metadata, np.ndarray):
            metadata = torch.from_numpy(metadata).float()
        
        # Ensure features and metadata are tensors
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(metadata, torch.Tensor):
            metadata = torch.tensor(metadata, dtype=torch.float32)
        
        # Move Sigma_inv to the same device as features if needed
        if self.Sigma_inv.device != features.device:
            self.Sigma_inv = self.Sigma_inv.to(features.device)
        
        # Move metadata to same device as features if needed
        if metadata.device != features.device:
            metadata = metadata.to(features.device)
        
        # Ensure Sigma_inv has no gradients
        if self.Sigma_inv.requires_grad:
            self.Sigma_inv = self.Sigma_inv.detach()
        
        # Compute beta = Sigma_inv @ (X^T @ f)
        XT_f = metadata.T @ features
        beta = self.Sigma_inv @ XT_f
        
        # Compute X @ beta
        X_beta = metadata @ beta
        
        # Return residual
        residual = features - X_beta
        
        return residual