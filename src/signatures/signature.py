"""Path Signature computation using Rough Path Theory.

The signature of a path is a sequence of iterated integrals that
uniquely characterizes the path (up to tree-like equivalence).
It provides a canonical feature set for sequential data that is:
1. Universal (can approximate any continuous function of paths)
2. Invariant to time reparametrization
3. Hierarchical (captures interactions at all orders)

References:
- Chevyrev & Kormilitzin, "A Primer on the Signature Method" (2016)
- Kidger & Lyons, "Signatory: Differentiable Computations of the Signature" (ICLR 2021)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Literal, Union
import logging

try:
    import signatory
    SIGNATORY_AVAILABLE = True
except ImportError:
    SIGNATORY_AVAILABLE = False

logger = logging.getLogger(__name__)


def signature_channels(dim: int, depth: int) -> int:
    """Compute the dimension of the signature at given depth.
    
    For a d-dimensional path, the signature up to depth k has dimension:
        Σ_{i=1}^{k} d^i = d(d^k - 1)/(d - 1)  if d > 1
        = k                                    if d = 1
    
    Args:
        dim: Path dimension
        depth: Signature truncation depth
        
    Returns:
        Total signature dimension
    """
    if dim == 1:
        return depth
    return dim * (dim ** depth - 1) // (dim - 1)


class SignatureFeatures:
    """Path signature feature extractor.
    
    Computes the truncated signature of paths for use as features
    in downstream ML models.
    
    Attributes:
        depth: Truncation depth for signature
        augmentations: List of path augmentations to apply
        basepoint: Whether to prepend a basepoint (usually origin)
    """
    
    def __init__(
        self,
        depth: int = 4,
        augmentations: List[str] = ["time"],
        basepoint: bool = True,
        normalize: bool = True
    ):
        if not SIGNATORY_AVAILABLE:
            raise ImportError("signatory required: pip install signatory")
        
        self.depth = depth
        self.augmentations = augmentations
        self.basepoint = basepoint
        self.normalize = normalize
        
        logger.info(
            f"Initialized SignatureFeatures: depth={depth}, "
            f"augmentations={augmentations}"
        )
    
    def _augment_path(self, path: NDArray) -> NDArray:
        """Apply path augmentations.
        
        Args:
            path: Path of shape (length, dim)
            
        Returns:
            Augmented path of shape (length, augmented_dim)
        """
        length, dim = path.shape
        augmented = [path]
        
        for aug in self.augmentations:
            if aug == "time":
                # Add time coordinate: (0, 1/n, 2/n, ..., 1)
                time_coord = np.linspace(0, 1, length).reshape(-1, 1)
                augmented.append(time_coord)
            
            elif aug == "leadlag":
                # Lead-lag transform: (x_t, x_{t-1})
                lagged = np.vstack([path[:1], path[:-1]])
                augmented.append(lagged)
            
            elif aug == "cumsum":
                # Cumulative sum (integrated path)
                augmented.append(np.cumsum(path, axis=0))
            
            elif aug == "returns":
                # Differences/returns
                diffs = np.vstack([np.zeros((1, dim)), np.diff(path, axis=0)])
                augmented.append(diffs)
            
            elif aug == "invisibility":
                # Invisibility reset for windowed computation
                # Adds dimension that goes 0 → 1 → 0 at boundaries
                invisible = np.zeros((length, 1))
                invisible[0] = 1
                invisible[-1] = 1
                augmented.append(invisible)
        
        return np.hstack(augmented)
    
    def _add_basepoint(self, path: NDArray) -> NDArray:
        """Add basepoint at origin.
        
        Adding a basepoint ensures the signature starts from a
        canonical reference point, improving stability.
        """
        basepoint = np.zeros((1, path.shape[1]))
        return np.vstack([basepoint, path])
    
    def transform(
        self,
        paths: Union[NDArray, List[NDArray]]
    ) -> NDArray:
        """Compute signature features for paths.
        
        Args:
            paths: Either array of shape (batch, length, dim) or
                   list of arrays of shape (length, dim)
                   
        Returns:
            Signature features of shape (batch, sig_dim)
        """
        # Handle list input
        if isinstance(paths, list):
            return np.vstack([self._transform_single(p) for p in paths])
        
        if paths.ndim == 2:
            paths = paths[np.newaxis, ...]
        
        batch_size = paths.shape[0]
        signatures = []
        
        for i in range(batch_size):
            sig = self._transform_single(paths[i])
            signatures.append(sig)
        
        return np.vstack(signatures)
    
    def _transform_single(self, path: NDArray) -> NDArray:
        """Compute signature for a single path."""
        # Augment path
        path_aug = self._augment_path(path)
        
        # Add basepoint
        if self.basepoint:
            path_aug = self._add_basepoint(path_aug)
        
        # Convert to torch and compute signature
        path_tensor = torch.from_numpy(path_aug).float().unsqueeze(0)
        sig = signatory.signature(path_tensor, depth=self.depth)
        sig_np = sig.numpy().squeeze()
        
        # Normalize by depth
        if self.normalize:
            sig_np = self._normalize_signature(sig_np, path_aug.shape[1])
        
        return sig_np
    
    def _normalize_signature(self, sig: NDArray, dim: int) -> NDArray:
        """Normalize signature components by factorial.
        
        The k-th level of the signature grows like O(length^k),
        so we normalize by k! to balance contributions.
        """
        normalized = sig.copy()
        
        start_idx = 0
        for k in range(1, self.depth + 1):
            level_size = dim ** k
            end_idx = start_idx + level_size
            
            # Divide by k!
            factorial = np.math.factorial(k)
            normalized[start_idx:end_idx] /= factorial
            
            start_idx = end_idx
        
        return normalized
    
    def get_output_dim(self, input_dim: int) -> int:
        """Get output dimension for given input.
        
        Args:
            input_dim: Original path dimension
            
        Returns:
            Signature dimension after augmentations
        """
        augmented_dim = input_dim
        
        for aug in self.augmentations:
            if aug == "time":
                augmented_dim += 1
            elif aug == "leadlag":
                augmented_dim += input_dim
            elif aug in ["cumsum", "returns"]:
                augmented_dim += input_dim
            elif aug == "invisibility":
                augmented_dim += 1
        
        return signature_channels(augmented_dim, self.depth)


class SignatureLayer(nn.Module):
    """PyTorch module for differentiable signature computation.
    
    Can be used as a layer in neural networks for end-to-end training.
    """
    
    def __init__(
        self,
        depth: int = 4,
        stream: bool = False,
        basepoint: bool = True
    ):
        super().__init__()
        
        if not SIGNATORY_AVAILABLE:
            raise ImportError("signatory required")
        
        self.depth = depth
        self.stream = stream
        self.basepoint = basepoint
    
    def forward(self, path: Tensor) -> Tensor:
        """Compute signature of path.
        
        Args:
            path: Tensor of shape (batch, length, channels)
            
        Returns:
            Signature tensor of shape:
                - (batch, sig_dim) if stream=False
                - (batch, length-1, sig_dim) if stream=True
        """
        if self.basepoint:
            # Prepend zeros as basepoint
            basepoint = torch.zeros_like(path[:, :1, :])
            path = torch.cat([basepoint, path], dim=1)
        
        if self.stream:
            return signatory.signature(path, depth=self.depth, stream=True)
        else:
            return signatory.signature(path, depth=self.depth)


class SignatureRNN(nn.Module):
    """RNN enhanced with signature features at each step.
    
    Combines the expressiveness of signatures with the
    memory efficiency of RNNs for long sequences.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sig_depth: int = 3,
        window_size: int = 10,
        rnn_type: Literal["lstm", "gru"] = "lstm"
    ):
        super().__init__()
        
        self.window_size = window_size
        self.sig_layer = SignatureLayer(depth=sig_depth)
        
        # Signature dimension
        sig_dim = signature_channels(input_dim + 1, sig_depth)  # +1 for time
        
        # RNN
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim + sig_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
    
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None
    ) -> tuple:
        """Process sequence with windowed signatures.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Initial hidden state
            
        Returns:
            Tuple of (output, hidden_state)
        """
        batch, seq_len, input_dim = x.shape
        
        # Add time coordinate
        time = torch.linspace(0, 1, seq_len, device=x.device)
        time = time.view(1, -1, 1).expand(batch, -1, -1)
        x_time = torch.cat([x, time], dim=-1)
        
        # Compute windowed signatures
        signatures = []
        for t in range(seq_len):
            start = max(0, t - self.window_size + 1)
            window = x_time[:, start:t+1, :]
            
            if window.shape[1] < 2:
                # Pad for short windows
                window = torch.cat([window, window], dim=1)
            
            sig = self.sig_layer(window)
            signatures.append(sig)
        
        sigs = torch.stack(signatures, dim=1)
        
        # Concatenate with original features
        x_aug = torch.cat([x, sigs], dim=-1)
        
        return self.rnn(x_aug, hidden)







