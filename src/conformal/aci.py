"""Adaptive Conformal Inference for non-exchangeable time series.

Standard conformal prediction assumes exchangeability, which fails for
time series. ACI maintains coverage guarantees under distribution shift
by adaptively updating the miscoverage rate.

Reference: Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift" (NeurIPS 2021)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, List
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AdaptiveConformalInference:
    """Adaptive Conformal Inference (ACI) for time series.
    
    Maintains long-run coverage guarantee by adaptively adjusting
    the miscoverage rate based on observed coverage errors.
    
    Key Property:
        lim_{T→∞} (1/T) Σ_{t=1}^T 𝟙(Y_t ∉ C_t) = α  almost surely
    
    This is a weaker but achievable guarantee under distribution shift.
    
    Attributes:
        alpha: Target miscoverage rate
        gamma: Learning rate for alpha adaptation
        alpha_t: Current adaptive miscoverage rate
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        alpha_min: float = 0.01,
        alpha_max: float = 0.50
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Adaptive miscoverage rate (starts at target)
        self.alpha_t = alpha
        
        # History for diagnostics
        self.alpha_history: List[float] = [alpha]
        self.coverage_history: List[bool] = []
        self.score_history: List[float] = []
        
        logger.info(f"Initialized ACI: alpha={alpha}, gamma={gamma}")
    
    def reset(self) -> None:
        """Reset adaptive state."""
        self.alpha_t = self.alpha
        self.alpha_history = [self.alpha]
        self.coverage_history = []
        self.score_history = []
    
    def update(
        self,
        y_true: float,
        interval: Tuple[float, float]
    ) -> None:
        """Update adaptive miscoverage rate based on observed outcome.
        
        Uses online gradient descent on the miscoverage rate:
            α_{t+1} = α_t + γ(α - err_t)
        
        where err_t = 1 if y_t ∉ C_t, 0 otherwise.
        
        Args:
            y_true: True observed value
            interval: Prediction interval (lower, upper)
        """
        lower, upper = interval
        
        # Check if covered
        covered = lower <= y_true <= upper
        err_t = 1.0 - float(covered)
        
        # Gradient update
        # If we covered (err=0) and alpha > target, decrease alpha (tighter intervals)
        # If we missed (err=1) and alpha < 1, increase alpha (wider intervals)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        
        # Clip to valid range
        self.alpha_t = np.clip(self.alpha_t, self.alpha_min, self.alpha_max)
        
        # Record history
        self.alpha_history.append(self.alpha_t)
        self.coverage_history.append(covered)
    
    def get_quantile_level(self) -> float:
        """Get current quantile level for interval construction.
        
        Returns:
            Quantile level (1 - α_t) for the prediction interval
        """
        return 1 - self.alpha_t
    
    def get_coverage_stats(self, window: Optional[int] = None) -> dict:
        """Get coverage statistics.
        
        Args:
            window: Rolling window size (None for all history)
            
        Returns:
            Dictionary with coverage statistics
        """
        if not self.coverage_history:
            return {"coverage": None, "n_samples": 0}
        
        if window is not None:
            recent = self.coverage_history[-window:]
        else:
            recent = self.coverage_history
        
        return {
            "coverage": np.mean(recent),
            "n_samples": len(recent),
            "target": 1 - self.alpha,
            "current_alpha": self.alpha_t,
            "coverage_gap": np.mean(recent) - (1 - self.alpha)
        }


class OnlineQuantileTracker:
    """Online quantile estimation for streaming data.
    
    Maintains running quantile estimates using the P² algorithm
    (Jain & Chlamtac, 1985) for memory-efficient streaming updates.
    """
    
    def __init__(self, quantile: float = 0.9):
        self.p = quantile
        self.n = 0
        
        # Marker positions
        self.q = np.zeros(5)
        # Marker heights
        self.n_pos = np.zeros(5)
        # Desired positions
        self.n_prime = np.zeros(5)
        # Desired position increments
        self.dn = np.array([0, self.p/2, self.p, (1+self.p)/2, 1])
    
    def update(self, x: float) -> None:
        """Update quantile estimate with new observation.
        
        Args:
            x: New observation
        """
        self.n += 1
        
        if self.n <= 5:
            self.q[self.n - 1] = x
            if self.n == 5:
                self.q.sort()
                self.n_pos = np.arange(1, 6)
            return
        
        # Find cell for new observation
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            k = np.searchsorted(self.q, x) - 1
            k = max(0, min(k, 3))
        
        # Increment positions above k
        self.n_pos[k+1:] += 1
        
        # Update desired positions
        self.n_prime = 1 + self.dn * (self.n - 1)
        
        # Adjust heights using P² formula
        for i in range(1, 4):
            d = self.n_prime[i] - self.n_pos[i]
            if (d >= 1 and self.n_pos[i+1] - self.n_pos[i] > 1) or \
               (d <= -1 and self.n_pos[i-1] - self.n_pos[i] < -1):
                d_sign = 1 if d >= 0 else -1
                
                # Parabolic formula
                qi_new = self._parabolic(i, d_sign)
                
                if self.q[i-1] < qi_new < self.q[i+1]:
                    self.q[i] = qi_new
                else:
                    # Linear formula
                    self.q[i] = self._linear(i, d_sign)
                
                self.n_pos[i] += d_sign
    
    def _parabolic(self, i: int, d: int) -> float:
        """Parabolic interpolation formula."""
        qi = self.q[i]
        qi_minus = self.q[i-1]
        qi_plus = self.q[i+1]
        ni = self.n_pos[i]
        ni_minus = self.n_pos[i-1]
        ni_plus = self.n_pos[i+1]
        
        return qi + d / (ni_plus - ni_minus) * (
            (ni - ni_minus + d) * (qi_plus - qi) / (ni_plus - ni) +
            (ni_plus - ni - d) * (qi - qi_minus) / (ni - ni_minus)
        )
    
    def _linear(self, i: int, d: int) -> float:
        """Linear interpolation formula."""
        return self.q[i] + d * (self.q[i+d] - self.q[i]) / (self.n_pos[i+d] - self.n_pos[i])
    
    @property
    def quantile(self) -> float:
        """Current quantile estimate."""
        if self.n < 5:
            if self.n == 0:
                return 0.0
            sorted_q = np.sort(self.q[:self.n])
            idx = int(self.p * (self.n - 1))
            return sorted_q[idx]
        return self.q[2]


class ACIWithScoreBuffer:
    """ACI with rolling buffer of nonconformity scores.
    
    Maintains a buffer of recent scores for quantile estimation,
    combined with adaptive alpha for long-run coverage.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.01,
        buffer_size: int = 500
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.buffer_size = buffer_size
        
        self.alpha_t = alpha
        self.score_buffer: deque = deque(maxlen=buffer_size)
        
        # Online quantile tracker as backup
        self.quantile_tracker = OnlineQuantileTracker(quantile=1-alpha)
    
    def add_score(self, score: float, covered: bool) -> None:
        """Add new nonconformity score and update.
        
        Args:
            score: Nonconformity score for this observation
            covered: Whether the true value was covered
        """
        self.score_buffer.append(score)
        self.quantile_tracker.update(score)
        
        # Update adaptive alpha
        err_t = 1.0 - float(covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.50)
    
    def get_threshold(self) -> float:
        """Get current conformal threshold.
        
        Uses buffer quantile at adaptive alpha level.
        """
        if len(self.score_buffer) < 10:
            return self.quantile_tracker.quantile
        
        scores = np.array(self.score_buffer)
        quantile_level = 1 - self.alpha_t
        
        # Finite-sample correction
        n = len(scores)
        adjusted_level = min(np.ceil((n + 1) * quantile_level) / n, 1.0)
        
        return np.quantile(scores, adjusted_level)
    
    def predict_interval(
        self,
        base_lower: float,
        base_upper: float
    ) -> Tuple[float, float]:
        """Construct ACI-adjusted prediction interval.
        
        Args:
            base_lower: Base model's lower prediction
            base_upper: Base model's upper prediction
            
        Returns:
            Adjusted (lower, upper) interval
        """
        threshold = self.get_threshold()
        return (base_lower - threshold, base_upper + threshold)







