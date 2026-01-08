"""
Eigensolve Utilities for SpectralDiffusion

Power iteration for efficient eigenvector computation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def power_iteration(
    matrix: torch.Tensor,
    k: int = 1,
    num_iters: int = 50,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k eigenvectors via power iteration with deflation.
    
    Args:
        matrix: [B, N, N] symmetric matrix
        k: Number of eigenvectors to compute
        num_iters: Maximum iterations per eigenvector
        tol: Convergence tolerance
        
    Returns:
        eigenvalues: [B, k] top-k eigenvalues
        eigenvectors: [B, N, k] corresponding eigenvectors
    """
    B, N, _ = matrix.shape
    device = matrix.device
    dtype = matrix.dtype
    
    eigenvectors = []
    eigenvalues = []
    
    # Work with a copy for deflation
    A = matrix.clone()
    
    for i in range(k):
        # Random initialization
        v = torch.randn(B, N, 1, device=device, dtype=dtype)
        v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        
        prev_eigenvalue = None
        
        for iteration in range(num_iters):
            # Power iteration step: v_new = A @ v
            v_new = torch.bmm(A, v)  # [B, N, 1]
            
            # Compute Rayleigh quotient (eigenvalue estimate)
            eigenvalue = torch.bmm(v.transpose(1, 2), v_new).squeeze()  # [B]
            
            # Normalize
            v_new = v_new / (torch.norm(v_new, dim=1, keepdim=True) + 1e-8)
            
            # Check convergence
            if prev_eigenvalue is not None:
                diff = (eigenvalue - prev_eigenvalue).abs().max()
                if diff < tol:
                    break
            
            prev_eigenvalue = eigenvalue
            v = v_new
        
        eigenvectors.append(v.squeeze(-1))  # [B, N]
        eigenvalues.append(eigenvalue)  # [B]
        
        # Deflation: A = A - lambda * v @ v^T
        A = A - eigenvalue.view(B, 1, 1) * torch.bmm(v, v.transpose(1, 2))
    
    eigenvectors = torch.stack(eigenvectors, dim=-1)  # [B, N, k]
    eigenvalues = torch.stack(eigenvalues, dim=-1)  # [B, k]
    
    return eigenvalues, eigenvectors


def compute_laplacian_eigenvectors(
    features: torch.Tensor,
    k: int,
    k_neighbors: int = 20,
    sigma: float = 1.0,
    num_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute eigenvectors of normalized Laplacian from features.
    
    Args:
        features: [B, N, D] feature vectors
        k: Number of eigenvectors
        k_neighbors: Number of neighbors for affinity
        sigma: Gaussian kernel bandwidth
        num_iters: Power iteration iterations
        
    Returns:
        eigenvalues: [B, k] smallest non-trivial eigenvalues
        eigenvectors: [B, N, k] corresponding eigenvectors
    """
    B, N, D = features.shape
    device = features.device
    dtype = features.dtype
    
    # Compute pairwise distances
    dists = torch.cdist(features, features)  # [B, N, N]
    
    # k-NN graph
    k_actual = min(k_neighbors, N - 1)
    _, indices = torch.topk(-dists, k=k_actual + 1, dim=-1)
    indices = indices[:, :, 1:]  # Remove self
    
    # Build sparse affinity with Gaussian kernel
    knn_dists = torch.gather(dists, 2, indices)  # [B, N, k]
    weights = torch.exp(-knn_dists / (2 * sigma ** 2))
    
    # Full affinity matrix
    affinity = torch.zeros(B, N, N, device=device, dtype=dtype)
    affinity.scatter_(2, indices, weights)
    affinity = (affinity + affinity.transpose(1, 2)) / 2  # Symmetrize
    
    # Compute normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    degree = affinity.sum(dim=-1)  # [B, N]
    d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    
    # D^{-1/2} @ W @ D^{-1/2}
    norm_affinity = d_inv_sqrt.unsqueeze(-1) * affinity * d_inv_sqrt.unsqueeze(-2)
    
    # L = I - norm_affinity
    identity = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    laplacian = identity - norm_affinity
    
    # For smallest eigenvalues of L, compute largest of (2I - L)
    shifted = 2 * identity - laplacian
    
    eigenvalues, eigenvectors = power_iteration(shifted, k=k, num_iters=num_iters)
    
    # Convert eigenvalues back: lambda_L = 2 - lambda_shifted
    eigenvalues = 2 - eigenvalues
    
    return eigenvalues, eigenvectors


def spectral_clustering_labels(
    features: torch.Tensor,
    k: int,
    k_neighbors: int = 20,
) -> torch.Tensor:
    """
    Compute spectral clustering labels.
    
    Args:
        features: [B, N, D] feature vectors
        k: Number of clusters
        k_neighbors: Number of neighbors for affinity
        
    Returns:
        labels: [B, N] cluster labels
    """
    B, N, D = features.shape
    
    # Get eigenvectors
    _, eigenvectors = compute_laplacian_eigenvectors(
        features, k=k, k_neighbors=k_neighbors
    )  # [B, N, k]
    
    # Normalize rows
    eigenvectors = F.normalize(eigenvectors, p=2, dim=-1)
    
    # K-means on eigenvector space
    # Simple implementation: assign to nearest centroid
    labels = torch.zeros(B, N, dtype=torch.long, device=features.device)
    
    for b in range(B):
        # K-means++ initialization
        centroids = [eigenvectors[b, torch.randint(0, N, (1,))]]
        
        for _ in range(k - 1):
            # Distances to nearest centroid
            dists = torch.stack([
                torch.norm(eigenvectors[b] - c.unsqueeze(0), dim=-1)
                for c in centroids
            ], dim=-1).min(dim=-1)[0]
            
            # Sample proportional to distance²
            probs = dists ** 2
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            centroids.append(eigenvectors[b, idx])
        
        centroids = torch.stack(centroids, dim=0).squeeze(1)  # [k, D]
        
        # Assign labels
        dists = torch.cdist(eigenvectors[b:b+1], centroids.unsqueeze(0)).squeeze(0)
        labels[b] = dists.argmin(dim=-1)
    
    return labels
