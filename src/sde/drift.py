"""Drift networks for Neural SDEs with stability guarantees.

Implements drift functions with various architectural choices and 
Lipschitz constraints for guaranteed existence/uniqueness of solutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional
import math


class DriftNetwork(nn.Module):
    """Standard MLP drift network with optional Lipschitz constraint.
    
    f(x, t) maps (state, time) → drift vector.
    
    Attributes:
        input_dim: Input dimension (state_dim + 1 for time)
        output_dim: Output dimension (state_dim)
        hidden_dims: List of hidden layer dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "silu",
        final_activation: Optional[str] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        activations = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "elu": nn.ELU
        }
        
        act_fn = activations[activation]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if final_activation is not None:
            layers.append(activations[final_activation]())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with orthogonal initialization for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute drift.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Drift of shape (batch, output_dim)
        """
        return self.net(x)


class LipschitzDrift(nn.Module):
    """Drift network with enforced Lipschitz constraint.
    
    Lipschitz continuity is crucial for SDE solution existence.
    Uses spectral normalization to bound the Lipschitz constant.
    
    |f(x) - f(y)| ≤ L|x - y| for all x, y
    
    This guarantees strong solutions exist and are unique.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        lipschitz_bound: float = 1.0,
        n_power_iterations: int = 1
    ):
        super().__init__()
        
        self.lipschitz_bound = lipschitz_bound
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            # Apply spectral normalization
            layer = nn.utils.spectral_norm(
                nn.Linear(prev_dim, h_dim),
                n_power_iterations=n_power_iterations
            )
            layers.append(layer)
            # GroupSort activation (1-Lipschitz)
            layers.append(GroupSort(num_groups=h_dim // 2))
            prev_dim = h_dim
        
        # Final layer with spectral norm
        layers.append(
            nn.utils.spectral_norm(
                nn.Linear(prev_dim, output_dim),
                n_power_iterations=n_power_iterations
            )
        )
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute Lipschitz-bounded drift.
        
        Args:
            x: Input tensor
            
        Returns:
            Drift tensor scaled by Lipschitz bound
        """
        return self.lipschitz_bound * self.net(x)


class GroupSort(nn.Module):
    """GroupSort activation function (1-Lipschitz).
    
    Sorts elements within groups, preserving gradient flow
    while maintaining Lipschitz constraint.
    
    Reference: Anil et al., "Sorting Out Lipschitz Function Approximation" (ICML 2019)
    """
    
    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        # Reshape to (batch, num_groups, group_size)
        x = x.reshape(batch_size, self.num_groups, -1)
        # Sort within each group
        x, _ = torch.sort(x, dim=-1)
        # Reshape back
        return x.reshape(batch_size, -1)


class ResidualDrift(nn.Module):
    """Residual drift network for learning perturbations to base dynamics.
    
    f(x, t) = f_base(x) + ε * f_residual(x, t)
    
    where f_base captures known dynamics and f_residual learns corrections.
    Useful when incorporating domain knowledge (e.g., mean-reversion).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_drift: Optional[nn.Module] = None,
        residual_scale: float = 0.1,
        hidden_dims: List[int] = [64, 64]
    ):
        super().__init__()
        
        self.residual_scale = residual_scale
        
        # Default base drift: mean-reversion
        if base_drift is None:
            self.base_drift = MeanReversionDrift(output_dim)
        else:
            self.base_drift = base_drift
        
        # Residual network
        self.residual = DriftNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            final_activation="tanh"  # Bounded output
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute residual-augmented drift.
        
        Args:
            x: Input (state, time) tensor
            
        Returns:
            Total drift
        """
        # Split state and time
        state = x[:, :-1]
        
        base = self.base_drift(state)
        residual = self.residual(x)
        
        return base + self.residual_scale * residual


class MeanReversionDrift(nn.Module):
    """Ornstein-Uhlenbeck mean-reversion drift.
    
    f(x) = θ(μ - x)
    
    where θ is the speed of mean-reversion and μ is the long-term mean.
    """
    
    def __init__(
        self,
        dim: int,
        theta: float = 1.0,
        learnable: bool = True
    ):
        super().__init__()
        
        if learnable:
            self.theta = nn.Parameter(torch.full((dim,), theta))
            self.mu = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("theta", torch.full((dim,), theta))
            self.register_buffer("mu", torch.zeros(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute mean-reversion drift.
        
        Args:
            x: State tensor of shape (batch, dim)
            
        Returns:
            Drift toward mean
        """
        return self.theta * (self.mu - x)


class TimeVaryingDrift(nn.Module):
    """Drift with explicit time-dependence via Fourier features.
    
    Encodes time using random Fourier features for smooth
    periodic patterns (e.g., intraday seasonality).
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 128],
        n_fourier_features: int = 16,
        max_freq: float = 10.0
    ):
        super().__init__()
        
        self.n_fourier_features = n_fourier_features
        
        # Random frequencies for Fourier features
        self.register_buffer(
            "freqs",
            torch.rand(n_fourier_features) * max_freq
        )
        
        # Network input: state + sin/cos features
        input_dim = state_dim + 2 * n_fourier_features
        
        self.net = DriftNetwork(
            input_dim=input_dim,
            output_dim=state_dim,
            hidden_dims=hidden_dims
        )
    
    def _time_embedding(self, t: Tensor) -> Tensor:
        """Compute Fourier time embedding.
        
        Args:
            t: Time tensor of shape (batch, 1)
            
        Returns:
            Fourier features of shape (batch, 2 * n_fourier_features)
        """
        # Compute sin and cos of time * frequency
        angles = t * self.freqs * 2 * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute time-varying drift.
        
        Args:
            x: Concatenated (state, time) tensor
            
        Returns:
            Drift tensor
        """
        state, t = x[:, :-1], x[:, -1:]
        time_features = self._time_embedding(t)
        augmented = torch.cat([state, time_features], dim=-1)
        return self.net.net(augmented)







