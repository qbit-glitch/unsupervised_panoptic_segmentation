"""
Multi-Scale Spectral Initialization for SpectralDiffusion

Implements principled slot initialization using spectral graph theory:
1. Compute sparse k-NN affinity matrix from DINOv3 features
2. Build normalized Laplacian
3. Extract top-K eigenvectors via power iteration
4. K-means++ initialization on eigenvector space

Based on spectral clustering theory (Ng et al. 2001, Von Luxburg 2007).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from einops import rearrange
import math


class MultiScaleSpectralInit(nn.Module):
    """
    Multi-scale spectral initialization for slot attention.
    
    Uses hierarchical spectral decomposition at multiple resolutions
    to initialize slot prototypes with principled object discovery.
    
    Args:
        scales: List of spatial scales for spectral decomposition
        slots_per_scale: Number of slots to extract per scale
        feature_dim: Input feature dimension
        k_neighbors: Number of neighbors for k-NN affinity
        sigma: Gaussian kernel bandwidth
        num_power_iters: Number of power iteration steps
        use_kmeans_pp: Whether to use K-means++ for prototype selection
    """
    
    def __init__(
        self,
        scales: List[int] = [8, 16, 32],
        slots_per_scale: int = 4,
        feature_dim: int = 768,
        k_neighbors: int = 20,
        sigma: float = 1.0,
        num_power_iters: int = 50,
        use_kmeans_pp: bool = True,
    ):
        super().__init__()
        
        self.scales = scales
        self.slots_per_scale = slots_per_scale
        self.feature_dim = feature_dim
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.num_power_iters = num_power_iters
        self.use_kmeans_pp = use_kmeans_pp
        
        # Total number of slots
        self.num_slots = len(scales) * slots_per_scale
        
        # Learnable slot projection (optional refinement)
        self.slot_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        
        # For learned baseline comparison (ablation A1)
        self.learned_slots = nn.Parameter(
            torch.randn(1, self.num_slots, feature_dim) * 0.02
        )
    
    def compute_pairwise_distances(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise L2 distances.
        
        Args:
            features: [B, N, D] feature vectors
            
        Returns:
            distances: [B, N, N] pairwise distances
        """
        # Efficient computation using expanded form
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        feat_sq = (features ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        
        # [B, N, N] = [B, N, 1] + [B, 1, N] - 2 * [B, N, N]
        distances = feat_sq + feat_sq.transpose(1, 2) - 2 * torch.bmm(
            features, features.transpose(1, 2)
        )
        
        # Clamp to avoid numerical issues
        distances = torch.clamp(distances, min=0.0)
        
        return distances
    
    def build_knn_affinity(
        self,
        features: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build sparse k-NN affinity matrix with Gaussian kernel.
        
        Args:
            features: [B, N, D] feature vectors
            k: Number of neighbors (default: self.k_neighbors)
            
        Returns:
            affinity: [B, N, N] affinity matrix (sparse-ish via masking)
            indices: [B, N, k] k-NN indices
        """
        B, N, D = features.shape
        k = k or min(self.k_neighbors, N - 1)
        
        # Compute pairwise distances
        distances = self.compute_pairwise_distances(features)  # [B, N, N]
        
        # Find k nearest neighbors
        # Use negative distances for topk (want smallest distances)
        _, indices = torch.topk(-distances, k=k + 1, dim=-1)  # [B, N, k+1]
        indices = indices[:, :, 1:]  # Remove self (first neighbor)
        
        # Build sparse affinity using Gaussian kernel
        # Gather k-NN distances
        knn_distances = torch.gather(
            distances,
            dim=2,
            index=indices,
        )  # [B, N, k]
        
        # Gaussian kernel weights
        weights = torch.exp(-knn_distances / (2 * self.sigma ** 2))  # [B, N, k]
        
        # Build full affinity matrix (sparse via masking)
        # Use float32 for numerical stability
        affinity = torch.zeros(B, N, N, device=features.device, dtype=torch.float32)
        
        # Scatter weights into affinity matrix
        row_idx = torch.arange(N, device=features.device).view(1, N, 1).expand(B, N, k)
        affinity.scatter_(2, indices, weights.float())
        
        # Make symmetric: W = (W + W^T) / 2
        affinity = (affinity + affinity.transpose(1, 2)) / 2
        
        return affinity, indices
    
    def compute_normalized_laplacian(
        self,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        
        Args:
            affinity: [B, N, N] affinity matrix
            
        Returns:
            laplacian: [B, N, N] normalized Laplacian
        """
        B, N, _ = affinity.shape
        
        # Compute degree: D_ii = sum_j W_ij
        degree = affinity.sum(dim=-1)  # [B, N]
        
        # D^{-1/2}
        d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)  # [B, N]
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        # D^{-1/2} W D^{-1/2}
        # = diag(d_inv_sqrt) @ W @ diag(d_inv_sqrt)
        normalized_affinity = d_inv_sqrt.unsqueeze(-1) * affinity * d_inv_sqrt.unsqueeze(-2)
        
        # L = I - normalized_affinity
        identity = torch.eye(N, device=affinity.device, dtype=affinity.dtype)
        identity = identity.unsqueeze(0).expand(B, -1, -1)
        
        laplacian = identity - normalized_affinity
        
        return laplacian
    
    def power_iteration(
        self,
        matrix: torch.Tensor,
        k: int,
        num_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute top-k eigenvectors via power iteration with deflation.
        
        For Laplacian, we want SMALLEST eigenvalues (after 0),
        so we compute largest eigenvectors of (I - L).
        
        Args:
            matrix: [B, N, N] symmetric matrix
            k: Number of eigenvectors
            num_iters: Number of iterations
            
        Returns:
            eigenvectors: [B, N, k] top-k eigenvectors
        """
        B, N, _ = matrix.shape
        num_iters = num_iters or self.num_power_iters
        
        # For Laplacian, compute eigenvectors of (2*I - L) = (I + normalized_affinity)
        # This gives eigenvectors corresponding to smallest eigenvalues of L
        # (excluding the trivial eigenvalue 0 with eigenvector 1)
        identity = torch.eye(N, device=matrix.device, dtype=matrix.dtype)
        identity = identity.unsqueeze(0).expand(B, -1, -1)
        
        # Shift matrix to find smallest eigenvalues
        shifted_matrix = 2 * identity - matrix
        
        eigenvectors = []
        
        for i in range(k):
            # Random initialization
            v = torch.randn(B, N, 1, device=matrix.device, dtype=matrix.dtype)
            v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
            
            for _ in range(num_iters):
                # Power iteration step
                v_new = torch.bmm(shifted_matrix, v)  # [B, N, 1]
                
                # Deflate previous eigenvectors (Gram-Schmidt)
                for ev in eigenvectors:
                    proj = torch.bmm(ev.transpose(1, 2), v_new)  # [B, 1, 1]
                    v_new = v_new - ev * proj
                
                # Normalize
                v_new = v_new / (torch.norm(v_new, dim=1, keepdim=True) + 1e-8)
                
                # Check convergence
                diff = torch.norm(v - v_new, dim=1).mean()
                v = v_new
                
                if diff < 1e-6:
                    break
            
            eigenvectors.append(v)
        
        # Stack eigenvectors: [B, N, k]
        eigenvectors = torch.cat(eigenvectors, dim=-1)
        
        return eigenvectors
    
    def kmeans_pp_init(
        self,
        features: torch.Tensor,
        eigenvectors: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        K-means++ initialization using eigenvector guidance.
        
        Args:
            features: [B, N, D] original features
            eigenvectors: [B, N, k_eig] eigenvectors for distance weighting
            k: Number of prototypes to select
            
        Returns:
            prototypes: [B, k, D] selected prototype features
        """
        B, N, D = features.shape
        device = features.device
        dtype = features.dtype
        
        prototypes = []
        
        for b in range(B):
            feat_b = features[b]  # [N, D]
            centers = []
            selected_indices = set()
            
            # First center: random
            idx = torch.randint(0, N, (1,), device=device).item()
            centers.append(feat_b[idx:idx+1])
            selected_indices.add(idx)
            
            for _ in range(k - 1):
                # Compute distances to nearest center
                centers_tensor = torch.cat(centers, dim=0)  # [num_centers, D]
                
                # Distance from each point to each center
                dists = torch.cdist(feat_b.unsqueeze(0), centers_tensor.unsqueeze(0)).squeeze(0)  # [N, num_centers]
                
                # Distance to nearest center
                min_dists = dists.min(dim=1)[0]  # [N]
                
                # Weight by eigenvector magnitude (focus on boundary points)
                if eigenvectors is not None:
                    eig_weight = torch.norm(eigenvectors[b], dim=-1)  # [N]
                    min_dists = min_dists * (1 + eig_weight)
                
                # Sample proportional to distance^2
                probs = min_dists ** 2
                
                # Add small epsilon to avoid zero probabilities
                probs = probs + 1e-8
                
                # Mask already selected points
                for sel_idx in selected_indices:
                    probs[sel_idx] = 0.0
                
                # Normalize
                prob_sum = probs.sum()
                if prob_sum <= 1e-8:
                    # Fall back to uniform sampling over unselected points
                    probs = torch.ones(N, device=device, dtype=dtype)
                    for sel_idx in selected_indices:
                        probs[sel_idx] = 0.0
                    probs = probs / probs.sum()
                else:
                    probs = probs / prob_sum
                
                # Sample next center
                idx = torch.multinomial(probs, 1).item()
                centers.append(feat_b[idx:idx+1])
                selected_indices.add(idx)
            
            prototypes.append(torch.cat(centers, dim=0))  # [k, D]
        
        prototypes = torch.stack(prototypes, dim=0)  # [B, k, D]
        return prototypes
    
    def spectral_init_single_scale(
        self,
        features: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Perform spectral initialization at a single scale.
        
        Args:
            features: [B, H, W, D] spatial features
            k: Number of prototypes to extract
            
        Returns:
            prototypes: [B, k, D] slot prototypes
        """
        B, H, W, D = features.shape
        N = H * W
        
        # Flatten spatial dimensions
        feat_flat = rearrange(features, 'b h w d -> b (h w) d')  # [B, N, D]
        
        # Build k-NN affinity
        k_nn = min(self.k_neighbors, N - 1)
        affinity, _ = self.build_knn_affinity(feat_flat, k=k_nn)
        
        # Compute normalized Laplacian
        laplacian = self.compute_normalized_laplacian(affinity)
        
        # Extract eigenvectors via power iteration
        # We want k eigenvectors for k-way partition
        eigenvectors = self.power_iteration(laplacian, k=k)  # [B, N, k]
        
        # K-means++ initialization in original feature space
        if self.use_kmeans_pp:
            prototypes = self.kmeans_pp_init(feat_flat, eigenvectors, k)
        else:
            # Simple: select points with extreme eigenvector values
            # For each eigenvector, pick the point with max absolute value
            prototypes = []
            for i in range(k):
                eig_i = eigenvectors[:, :, i].abs()  # [B, N]
                max_idx = eig_i.argmax(dim=1)  # [B]
                proto = torch.gather(
                    feat_flat,
                    dim=1,
                    index=max_idx.view(B, 1, 1).expand(-1, -1, D),
                ).squeeze(1)  # [B, D]
                prototypes.append(proto)
            prototypes = torch.stack(prototypes, dim=1)  # [B, k, D]
        
        return prototypes
    
    def forward(
        self,
        multiscale_features: dict,
        mode: str = "spectral",
    ) -> torch.Tensor:
        """
        Initialize slots from multi-scale features.
        
        Args:
            multiscale_features: Dict mapping scale -> [B, H, W, D] features
            mode: Initialization mode:
                - "spectral": Full spectral initialization (default)
                - "random": Random initialization (ablation A1)
                - "learned": Learnable slot embeddings (ablation A1)
                
        Returns:
            slots: [B, K, D] initialized slot embeddings
        """
        # Get batch size from first scale
        first_scale = self.scales[0]
        B = multiscale_features[first_scale].shape[0]
        device = multiscale_features[first_scale].device
        dtype = multiscale_features[first_scale].dtype
        
        if mode == "random":
            # Random initialization (ablation baseline)
            slots = torch.randn(
                B, self.num_slots, self.feature_dim,
                device=device, dtype=dtype,
            ) * 0.1
            return self.slot_projection(slots)
        
        elif mode == "learned":
            # Learnable slots (ablation baseline)
            slots = self.learned_slots.expand(B, -1, -1).to(dtype)
            return self.slot_projection(slots)
        
        elif mode == "spectral":
            # Full multi-scale spectral initialization
            all_prototypes = []
            
            for scale in self.scales:
                if scale not in multiscale_features:
                    raise ValueError(f"Missing features for scale {scale}")
                
                features = multiscale_features[scale]  # [B, H, W, D]
                
                # Spectral initialization at this scale
                prototypes = self.spectral_init_single_scale(
                    features, k=self.slots_per_scale
                )  # [B, slots_per_scale, D]
                
                all_prototypes.append(prototypes)
            
            # Concatenate multi-scale prototypes
            slots = torch.cat(all_prototypes, dim=1)  # [B, num_slots, D]
            
            # Apply projection
            slots = self.slot_projection(slots)
            
            return slots
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_spectral_masks(
        self,
        multiscale_features: dict,
    ) -> torch.Tensor:
        """
        Compute spectral clustering masks for consistency loss.
        
        Returns soft assignment masks based on eigenvector similarity.
        
        Args:
            multiscale_features: Dict mapping scale -> [B, H, W, D] features
            
        Returns:
            masks: [B, K, H*W] soft spectral masks at finest scale
        """
        # Use finest scale for mask computation
        finest_scale = min(self.scales)
        features = multiscale_features[finest_scale]  # [B, H, W, D]
        
        B, H, W, D = features.shape
        N = H * W
        
        # Flatten
        feat_flat = rearrange(features, 'b h w d -> b (h w) d')
        
        # Build affinity and Laplacian
        affinity, _ = self.build_knn_affinity(feat_flat)
        laplacian = self.compute_normalized_laplacian(affinity)
        
        # Get eigenvectors
        eigenvectors = self.power_iteration(laplacian, k=self.num_slots)  # [B, N, K]
        
        # Convert to soft masks via softmax over eigenvector magnitudes
        # Each eigenvector roughly corresponds to a cluster
        soft_masks = F.softmax(eigenvectors.abs() * 10, dim=-1)  # [B, N, K]
        
        # Transpose to [B, K, N]
        soft_masks = soft_masks.transpose(1, 2)
        
        return soft_masks


if __name__ == "__main__":
    # Test spectral initialization
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    # Create module
    spectral_init = MultiScaleSpectralInit(
        scales=[8, 16, 32],
        slots_per_scale=4,
        feature_dim=768,
        k_neighbors=20,
    ).to(device)
    
    print(f"Total slots: {spectral_init.num_slots}")
    
    # Create dummy multi-scale features
    B = 2
    multiscale_features = {
        8: torch.randn(B, 8, 8, 768, device=device),
        16: torch.randn(B, 16, 16, 768, device=device),
        32: torch.randn(B, 32, 32, 768, device=device),
    }
    
    # Test spectral initialization
    print("\nTesting spectral initialization...")
    slots = spectral_init(multiscale_features, mode="spectral")
    print(f"Spectral slots: {slots.shape}")
    
    # Test random initialization
    print("\nTesting random initialization...")
    slots_random = spectral_init(multiscale_features, mode="random")
    print(f"Random slots: {slots_random.shape}")
    
    # Test learned initialization
    print("\nTesting learned initialization...")
    slots_learned = spectral_init(multiscale_features, mode="learned")
    print(f"Learned slots: {slots_learned.shape}")
    
    # Test spectral masks
    print("\nTesting spectral masks...")
    masks = spectral_init.get_spectral_masks(multiscale_features)
    print(f"Spectral masks: {masks.shape}")
