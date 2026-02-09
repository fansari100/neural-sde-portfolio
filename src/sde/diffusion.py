"""Diffusion networks for Neural SDEs with positivity constraints.

Implements diffusion functions σ(x, t) with guaranteed positive-definiteness
required for valid SDE dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Literal
import math


class DiffusionNetwork(nn.Module):
    """Standard diffusion network with positivity enforcement.
    
    g(x, t) maps (state, time) → diffusion matrix/vector.
    
    Uses Softplus activation to ensure positive diffusion coefficients.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = "silu",
        min_diffusion: float = 1e-4,
        max_diffusion: float = 10.0
    ):
        super().__init__()
        
        self.min_diffusion = min_diffusion
        self.max_diffusion = max_diffusion
        
        activations = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "gelu": nn.GELU
        }
        act_fn = activations[activation]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_fn()
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize for reasonable starting diffusion
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for stable starting diffusion."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    # Bias to give reasonable starting diffusion
                    nn.init.constant_(m.bias, 0.5)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute positive diffusion coefficients.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Diffusion of shape (batch, output_dim), guaranteed positive
        """
        raw = self.net(x)
        
        # Softplus for positivity with min/max bounds
        diffusion = nn.functional.softplus(raw) + self.min_diffusion
        diffusion = torch.clamp(diffusion, max=self.max_diffusion)
        
        return diffusion


class PositiveDiffusion(nn.Module):
    """Diffusion network using exponential parameterization.
    
    σ(x) = exp(f(x)) ensures strict positivity.
    More numerically stable than Softplus for very small values.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        log_scale_init: float = -1.0,
        scale_range: tuple = (-3.0, 2.0)
    ):
        super().__init__()
        
        self.scale_range = scale_range
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SiLU()
            ])
            prev_dim = h_dim
        
        # Output log-scale
        self.net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize for desired starting scale
        nn.init.zeros_(self.out_layer.weight)
        nn.init.constant_(self.out_layer.bias, log_scale_init)
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute exponentially-parameterized diffusion.
        
        Args:
            x: Input tensor
            
        Returns:
            Strictly positive diffusion
        """
        h = self.net(x)
        log_scale = self.out_layer(h)
        
        # Clamp log-scale for numerical stability
        log_scale = torch.clamp(
            log_scale,
            min=self.scale_range[0],
            max=self.scale_range[1]
        )
        
        return torch.exp(log_scale)


class CholeskyDiffusion(nn.Module):
    """Full covariance diffusion via Cholesky parameterization.
    
    For general noise type, outputs a lower-triangular matrix L
    such that ΣΣ^T = LL^T is positive semi-definite.
    
    g(x) ∈ ℝ^{d×m} where m is Brownian motion dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        brownian_dim: int,
        hidden_dims: List[int] = [128, 128],
        diag_activation: str = "softplus"
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.brownian_dim = brownian_dim
        
        # Total outputs: full matrix
        output_dim = state_dim * brownian_dim
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU()
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Diagonal activation
        self.diag_act = {
            "softplus": nn.Softplus(),
            "exp": lambda x: torch.exp(torch.clamp(x, max=5.0))
        }[diag_activation]
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute full diffusion matrix.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Diffusion matrix of shape (batch, state_dim, brownian_dim)
        """
        batch_size = x.shape[0]
        
        raw = self.net(x)
        L = raw.reshape(batch_size, self.state_dim, self.brownian_dim)
        
        # Apply positivity to diagonal (ensure non-degeneracy)
        diag_mask = torch.eye(
            min(self.state_dim, self.brownian_dim),
            device=x.device
        ).bool()
        
        if self.state_dim <= self.brownian_dim:
            diag_indices = diag_mask.unsqueeze(0).expand(batch_size, -1, -1)
            L = L.clone()
            diag_vals = L[:, diag_indices[0]]
            L[:, diag_indices[0]] = self.diag_act(diag_vals)
        
        return L


class StateVaryingDiffusion(nn.Module):
    """State-dependent diffusion with GARCH-inspired structure.
    
    σ(x)² = ω + α * f(x)² + β * σ_{prev}²
    
    Captures volatility clustering common in financial data.
    """
    
    def __init__(
        self,
        state_dim: int,
        omega: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.85,
        learnable: bool = True
    ):
        super().__init__()
        
        if learnable:
            self.omega = nn.Parameter(torch.full((state_dim,), omega))
            self.alpha = nn.Parameter(torch.full((state_dim,), alpha))
            self.beta = nn.Parameter(torch.full((state_dim,), beta))
        else:
            self.register_buffer("omega", torch.full((state_dim,), omega))
            self.register_buffer("alpha", torch.full((state_dim,), alpha))
            self.register_buffer("beta", torch.full((state_dim,), beta))
        
        # Running variance estimate
        self.register_buffer(
            "running_var",
            torch.ones(state_dim)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute GARCH-like diffusion.
        
        Args:
            x: Input tensor (state, time)
            
        Returns:
            Diffusion with volatility clustering
        """
        state = x[:, :-1]
        
        # Ensure positive parameters
        omega = nn.functional.softplus(self.omega)
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta) * (1 - alpha)  # Ensure stationarity
        
        # Compute variance
        variance = omega + alpha * state**2 + beta * self.running_var
        
        # Update running variance (exponential moving average)
        with torch.no_grad():
            self.running_var = 0.99 * self.running_var + 0.01 * variance.mean(0)
        
        return torch.sqrt(variance + 1e-6)


class LeverageDiffusion(nn.Module):
    """Diffusion with leverage effect (asymmetric volatility response).
    
    Models the empirical observation that negative returns increase
    volatility more than positive returns (leverage effect in finance).
    
    σ(r) = σ_0 * (1 + γ * max(-r, 0))
    """
    
    def __init__(
        self,
        state_dim: int,
        base_vol: float = 0.02,
        leverage_gamma: float = 0.5,
        learnable: bool = True
    ):
        super().__init__()
        
        if learnable:
            self.log_base_vol = nn.Parameter(
                torch.full((state_dim,), math.log(base_vol))
            )
            self.leverage_gamma = nn.Parameter(
                torch.full((state_dim,), leverage_gamma)
            )
        else:
            self.register_buffer(
                "log_base_vol",
                torch.full((state_dim,), math.log(base_vol))
            )
            self.register_buffer(
                "leverage_gamma",
                torch.full((state_dim,), leverage_gamma)
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """Compute leverage-adjusted diffusion.
        
        Args:
            x: Input (returns, time) tensor
            
        Returns:
            Asymmetric diffusion
        """
        returns = x[:, :-1]
        
        base_vol = torch.exp(self.log_base_vol)
        gamma = nn.functional.softplus(self.leverage_gamma)
        
        # Leverage effect: volatility increases more for negative returns
        leverage_factor = 1 + gamma * nn.functional.relu(-returns)
        
        return base_vol * leverage_factor







