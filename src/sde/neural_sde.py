"""Neural Stochastic Differential Equations with path-dependent dynamics.

Implements Neural SDEs where drift and diffusion are learned neural networks,
with optional path signature augmentation for capturing sequential dependencies.

References:
- Li et al., "Scalable Gradients for Stochastic Differential Equations" (AISTATS 2020)
- Kidger et al., "Neural SDEs as Infinite-Dimensional GANs" (ICML 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Literal, Tuple
import logging

try:
    import torchsde
    from torchsde import sdeint, sdeint_adjoint
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False

try:
    import signatory
    SIGNATORY_AVAILABLE = True
except ImportError:
    SIGNATORY_AVAILABLE = False

logger = logging.getLogger(__name__)


class NeuralSDE(nn.Module):
    """Neural SDE with learned drift and diffusion functions.
    
    Models continuous-time stochastic dynamics:
        dX_t = f_θ(X_t, t)dt + g_φ(X_t, t)dW_t
    
    where f_θ is the drift network and g_φ is the diffusion network.
    
    Attributes:
        state_dim: Dimension of the state space
        drift_net: Neural network for drift function μ(x, t)
        diffusion_net: Neural network for diffusion function σ(x, t)
        noise_type: Type of noise ("diagonal", "general", "scalar")
        sde_type: SDE interpretation ("ito", "stratonovich")
    """
    
    sde_type: str = "ito"
    noise_type: str = "diagonal"
    
    def __init__(
        self,
        state_dim: int,
        drift_net: nn.Module,
        diffusion_net: nn.Module,
        noise_type: Literal["diagonal", "general", "scalar"] = "diagonal",
        sde_type: Literal["ito", "stratonovich"] = "ito"
    ):
        super().__init__()
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde required: pip install torchsde")
        
        self.state_dim = state_dim
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.noise_type = noise_type
        self.sde_type = sde_type
        
        logger.info(
            f"Initialized NeuralSDE: state_dim={state_dim}, "
            f"noise={noise_type}, type={sde_type}"
        )
    
    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Drift function f(t, y) = μ(y, t).
        
        Args:
            t: Time tensor of shape ()
            y: State tensor of shape (batch, state_dim)
            
        Returns:
            Drift of shape (batch, state_dim)
        """
        # Concatenate time as additional feature
        t_expanded = t.expand(y.shape[0], 1)
        inputs = torch.cat([y, t_expanded], dim=-1)
        return self.drift_net(inputs)
    
    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Diffusion function g(t, y) = σ(y, t).
        
        Args:
            t: Time tensor of shape ()
            y: State tensor of shape (batch, state_dim)
            
        Returns:
            Diffusion of shape depending on noise_type:
                - diagonal: (batch, state_dim)
                - general: (batch, state_dim, brownian_dim)
                - scalar: (batch, 1)
        """
        t_expanded = t.expand(y.shape[0], 1)
        inputs = torch.cat([y, t_expanded], dim=-1)
        return self.diffusion_net(inputs)
    
    def sample(
        self,
        y0: Tensor,
        ts: Tensor,
        dt: float = 0.01,
        method: str = "milstein",
        adaptive: bool = False,
        adjoint: bool = False,
        **kwargs
    ) -> Tensor:
        """Sample trajectories from the Neural SDE.
        
        Args:
            y0: Initial state of shape (batch, state_dim)
            ts: Time points of shape (n_times,) at which to evaluate
            dt: Step size for fixed-step solvers
            method: SDE solver method ("euler", "milstein", "srk")
            adaptive: Whether to use adaptive stepping
            adjoint: Whether to use adjoint method for memory efficiency
            
        Returns:
            Trajectories of shape (n_times, batch, state_dim)
        """
        sdeint_fn = sdeint_adjoint if adjoint else sdeint
        
        return sdeint_fn(
            self,
            y0,
            ts,
            dt=dt,
            method=method,
            adaptive=adaptive,
            **kwargs
        )


class SignatureAugmentedSDE(nn.Module):
    """Neural SDE with path signature-augmented dynamics.
    
    The drift and diffusion depend on both current state and the
    signature of the historical path, enabling path-dependent dynamics
    that capture sequential structure.
    
    dX_t = f_θ(X_t, Sig(X_{[0,t]}))dt + g_φ(X_t, Sig(X_{[0,t]}))dW_t
    
    Attributes:
        state_dim: Dimension of the state space
        sig_depth: Depth of the path signature
        window_size: Size of the historical window for signature
    """
    
    sde_type: str = "ito"
    noise_type: str = "diagonal"
    
    def __init__(
        self,
        state_dim: int,
        sig_depth: int = 4,
        window_size: int = 20,
        hidden_dims: list = [128, 128],
        activation: str = "silu"
    ):
        super().__init__()
        
        if not SIGNATORY_AVAILABLE:
            raise ImportError("signatory required: pip install signatory")
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde required: pip install torchsde")
        
        self.state_dim = state_dim
        self.sig_depth = sig_depth
        self.window_size = window_size
        
        # Compute signature dimension
        self.sig_dim = signatory.signature_channels(
            channels=state_dim,
            depth=sig_depth
        )
        
        # Combined input dimension: state + signature
        input_dim = state_dim + self.sig_dim + 1  # +1 for time
        
        # Get activation
        act = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]
        
        # Drift network
        drift_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            drift_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                act()
            ])
            prev_dim = h_dim
        drift_layers.append(nn.Linear(prev_dim, state_dim))
        self.drift_net = nn.Sequential(*drift_layers)
        
        # Diffusion network (with Softplus for positivity)
        diff_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            diff_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                act()
            ])
            prev_dim = h_dim
        diff_layers.extend([
            nn.Linear(prev_dim, state_dim),
            nn.Softplus()  # Ensure positive diffusion
        ])
        self.diffusion_net = nn.Sequential(*diff_layers)
        
        # Path buffer for signature computation
        self.register_buffer("path_buffer", torch.zeros(1, window_size, state_dim))
        
        logger.info(
            f"Initialized SignatureAugmentedSDE: state_dim={state_dim}, "
            f"sig_depth={sig_depth}, sig_dim={self.sig_dim}"
        )
    
    def _compute_signature(self, path: Tensor) -> Tensor:
        """Compute path signature.
        
        Args:
            path: Path tensor of shape (batch, length, channels)
            
        Returns:
            Signature of shape (batch, sig_dim)
        """
        # Basepoint augmentation for better representation
        basepoint = torch.zeros_like(path[:, :1, :])
        path_aug = torch.cat([basepoint, path], dim=1)
        
        return signatory.signature(path_aug, depth=self.sig_depth)
    
    def update_path_buffer(self, new_state: Tensor) -> None:
        """Update the rolling path buffer with new state.
        
        Args:
            new_state: New state of shape (batch, state_dim)
        """
        batch_size = new_state.shape[0]
        
        # Expand buffer if needed
        if self.path_buffer.shape[0] != batch_size:
            self.path_buffer = self.path_buffer.expand(
                batch_size, -1, -1
            ).clone()
        
        # Shift and append
        self.path_buffer = torch.cat([
            self.path_buffer[:, 1:, :],
            new_state.unsqueeze(1)
        ], dim=1)
    
    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Signature-augmented drift function.
        
        Args:
            t: Time tensor
            y: State tensor of shape (batch, state_dim)
            
        Returns:
            Drift of shape (batch, state_dim)
        """
        # Update path buffer
        self.update_path_buffer(y)
        
        # Compute signature
        sig = self._compute_signature(self.path_buffer)
        
        # Concatenate state, signature, and time
        t_expanded = t.expand(y.shape[0], 1)
        inputs = torch.cat([y, sig, t_expanded], dim=-1)
        
        return self.drift_net(inputs)
    
    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Signature-augmented diffusion function.
        
        Args:
            t: Time tensor
            y: State tensor of shape (batch, state_dim)
            
        Returns:
            Diffusion of shape (batch, state_dim)
        """
        # Use cached signature from drift call
        sig = self._compute_signature(self.path_buffer)
        
        t_expanded = t.expand(y.shape[0], 1)
        inputs = torch.cat([y, sig, t_expanded], dim=-1)
        
        return self.diffusion_net(inputs)
    
    def reset_path_buffer(self, batch_size: int = 1) -> None:
        """Reset path buffer to zeros."""
        self.path_buffer = torch.zeros(
            batch_size, self.window_size, self.state_dim,
            device=self.path_buffer.device,
            dtype=self.path_buffer.dtype
        )


class LatentNeuralSDE(nn.Module):
    """Latent Neural SDE with encoder-decoder architecture.
    
    Maps observations to a latent space, evolves dynamics via Neural SDE,
    and decodes back to observation space. Useful for high-dimensional
    financial data like limit order books.
    
    Architecture:
        X → Encoder → z_0 → Neural SDE → z_t → Decoder → X̂_t
    """
    
    sde_type: str = "ito"
    noise_type: str = "diagonal"
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        encoder: nn.Module,
        decoder: nn.Module,
        drift_net: nn.Module,
        diffusion_net: nn.Module
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sde = NeuralSDE(
            state_dim=latent_dim,
            drift_net=drift_net,
            diffusion_net=diffusion_net
        )
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode observations to latent distribution.
        
        Args:
            x: Observations of shape (batch, obs_dim)
            
        Returns:
            Tuple of (mean, log_var) of latent distribution
        """
        h = self.encoder(x)
        mean, log_var = h.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent state to observations.
        
        Args:
            z: Latent state of shape (batch, latent_dim)
            
        Returns:
            Reconstructed observations of shape (batch, obs_dim)
        """
        return self.decoder(z)
    
    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Latent drift function."""
        return self.latent_sde.f(t, y)
    
    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Latent diffusion function."""
        return self.latent_sde.g(t, y)
    
    def forward(
        self,
        x0: Tensor,
        ts: Tensor,
        **sde_kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass: encode, evolve, decode.
        
        Args:
            x0: Initial observations of shape (batch, obs_dim)
            ts: Time points
            
        Returns:
            Tuple of (decoded_trajectories, mean, log_var)
        """
        # Encode initial state
        mean, log_var = self.encode(x0)
        z0 = self.reparameterize(mean, log_var)
        
        # Evolve in latent space
        z_traj = self.latent_sde.sample(z0, ts, **sde_kwargs)
        
        # Decode trajectories
        n_times, batch, _ = z_traj.shape
        z_flat = z_traj.reshape(-1, self.latent_dim)
        x_flat = self.decode(z_flat)
        x_traj = x_flat.reshape(n_times, batch, self.obs_dim)
        
        return x_traj, mean, log_var







