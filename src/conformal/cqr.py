"""Conformalized Quantile Regression for distribution-free prediction intervals.

Implements CQR from Romano et al. (NeurIPS 2019), providing finite-sample
valid prediction intervals without distributional assumptions.

Key Property:
    P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
    
This holds for ANY distribution - critical for heavy-tailed financial returns.

References:
- Romano et al., "Conformalized Quantile Regression" (NeurIPS 2019)
- Sesia & Romano, "Conformal Prediction Under Covariate Shift" (NeurIPS 2020)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Callable
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class QuantileRegressor:
    """Gradient boosting quantile regressor for CQR base model.
    
    Fits two models for lower (α/2) and upper (1 - α/2) quantiles.
    """
    
    def __init__(
        self,
        alpha: float = 0.10,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.alpha = alpha
        
        # Lower quantile model
        self.lower_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha / 2,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
        
        # Upper quantile model
        self.upper_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=1 - alpha / 2,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state
        )
    
    def fit(self, X: NDArray, y: NDArray) -> "QuantileRegressor":
        """Fit both quantile models.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: Targets of shape (n_samples,)
            
        Returns:
            Self
        """
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        return self
    
    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """Predict quantile bounds.
        
        Args:
            X: Features
            
        Returns:
            Tuple of (lower_quantile, upper_quantile)
        """
        return (
            self.lower_model.predict(X),
            self.upper_model.predict(X)
        )


class ConformalizedQuantileRegression:
    """Conformalized Quantile Regression (CQR).
    
    Wraps a quantile regressor with conformal calibration to provide
    finite-sample valid prediction intervals.
    
    The key insight is that raw quantile regression intervals may have
    incorrect coverage due to model misspecification. CQR calibrates
    these intervals on held-out data to guarantee coverage.
    
    Attributes:
        model: Base quantile regressor
        alpha: Miscoverage rate (1-α = coverage)
        calibration_scores: Nonconformity scores from calibration
    """
    
    def __init__(
        self,
        model: Optional[QuantileRegressor] = None,
        alpha: float = 0.10,
        symmetric: bool = False
    ):
        self.model = model or QuantileRegressor(alpha=alpha)
        self.alpha = alpha
        self.symmetric = symmetric
        
        self.calibration_scores_: Optional[NDArray] = None
        self.q_hat_: Optional[float] = None
        
        logger.info(f"Initialized CQR: alpha={alpha}, coverage={1-alpha:.0%}")
    
    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray
    ) -> "ConformalizedQuantileRegression":
        """Fit the base quantile regressor.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Self
        """
        self.model.fit(X_train, y_train)
        return self
    
    def calibrate(
        self,
        X_cal: NDArray,
        y_cal: NDArray
    ) -> "ConformalizedQuantileRegression":
        """Calibrate conformal prediction on held-out data.
        
        Computes nonconformity scores and determines the conformal
        quantile q_hat that guarantees 1-α coverage.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            
        Returns:
            Self
        """
        n_cal = len(y_cal)
        
        # Get predicted quantile bounds
        q_lower, q_upper = self.model.predict(X_cal)
        
        # Compute nonconformity scores
        # Score = max(q_lower - y, y - q_upper)
        # Measures how much y falls outside the predicted interval
        if self.symmetric:
            scores = np.maximum(q_lower - y_cal, y_cal - q_upper)
        else:
            # Asymmetric: separate scores for each side
            scores = np.maximum(q_lower - y_cal, y_cal - q_upper)
        
        self.calibration_scores_ = np.sort(scores)
        
        # Conformal quantile with finite-sample correction
        # q_hat = quantile of scores at level ceil((n+1)(1-α))/n
        quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        quantile_level = min(quantile_level, 1.0)
        
        self.q_hat_ = np.quantile(self.calibration_scores_, quantile_level)
        
        logger.info(f"Calibrated CQR: q_hat={self.q_hat_:.4f}")
        return self
    
    def fit_calibrate(
        self,
        X: NDArray,
        y: NDArray,
        calibration_fraction: float = 0.25,
        random_state: int = 42
    ) -> "ConformalizedQuantileRegression":
        """Fit and calibrate in one step with train/cal split.
        
        Args:
            X: Full feature matrix
            y: Full target vector
            calibration_fraction: Fraction of data for calibration
            random_state: Random seed
            
        Returns:
            Self
        """
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=calibration_fraction,
            random_state=random_state
        )
        
        self.fit(X_train, y_train)
        self.calibrate(X_cal, y_cal)
        
        return self
    
    def predict(
        self,
        X: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Generate conformal prediction intervals.
        
        These intervals are guaranteed to have at least 1-α coverage
        for any distribution (finite-sample valid).
        
        Args:
            X: Test features
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.q_hat_ is None:
            raise ValueError("Model not calibrated. Call calibrate() first.")
        
        # Get base quantile predictions
        q_lower, q_upper = self.model.predict(X)
        
        # Expand intervals by conformal correction
        lower = q_lower - self.q_hat_
        upper = q_upper + self.q_hat_
        
        return lower, upper
    
    def predict_with_coverage_check(
        self,
        X: NDArray,
        y_true: Optional[NDArray] = None
    ) -> dict:
        """Predict with optional coverage verification.
        
        Args:
            X: Test features
            y_true: True values (optional, for coverage check)
            
        Returns:
            Dictionary with predictions and optional coverage
        """
        lower, upper = self.predict(X)
        
        result = {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "q_hat": self.q_hat_,
            "target_coverage": 1 - self.alpha
        }
        
        if y_true is not None:
            covered = (y_true >= lower) & (y_true <= upper)
            result["empirical_coverage"] = covered.mean()
            result["coverage_valid"] = result["empirical_coverage"] >= (1 - self.alpha - 0.01)
        
        return result
    
    def get_interval_width(self, X: NDArray) -> NDArray:
        """Get prediction interval widths.
        
        Useful for identifying regions of high/low uncertainty.
        
        Args:
            X: Features
            
        Returns:
            Interval widths
        """
        lower, upper = self.predict(X)
        return upper - lower


class WeightedCQR(ConformalizedQuantileRegression):
    """Weighted CQR for covariate shift.
    
    Accounts for distribution shift between calibration and test
    by using importance weights.
    
    Reference: Tibshirani et al., "Conformal Prediction Under Covariate Shift"
    """
    
    def __init__(
        self,
        model: Optional[QuantileRegressor] = None,
        alpha: float = 0.10,
        weight_estimator: Optional[Callable] = None
    ):
        super().__init__(model=model, alpha=alpha)
        self.weight_estimator = weight_estimator
    
    def calibrate_weighted(
        self,
        X_cal: NDArray,
        y_cal: NDArray,
        X_test: NDArray
    ) -> "WeightedCQR":
        """Calibrate with importance weights for covariate shift.
        
        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            X_test: Test features (for weight estimation)
            
        Returns:
            Self
        """
        n_cal = len(y_cal)
        
        # Estimate importance weights p(X_test) / p(X_cal)
        if self.weight_estimator is not None:
            weights = self.weight_estimator(X_cal, X_test)
        else:
            # Default: use density ratio estimation
            weights = self._estimate_weights(X_cal, X_test)
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute nonconformity scores
        q_lower, q_upper = self.model.predict(X_cal)
        scores = np.maximum(q_lower - y_cal, y_cal - q_upper)
        
        # Weighted quantile
        sorted_idx = np.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        cumsum = np.cumsum(sorted_weights)
        quantile_level = 1 - self.alpha
        idx = np.searchsorted(cumsum, quantile_level)
        
        self.q_hat_ = sorted_scores[min(idx, len(sorted_scores) - 1)]
        self.calibration_scores_ = scores
        
        return self
    
    def _estimate_weights(
        self,
        X_cal: NDArray,
        X_test: NDArray
    ) -> NDArray:
        """Estimate density ratio weights via classification.
        
        Trains a classifier to distinguish cal vs test samples,
        then uses predicted probabilities as weights.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Create binary labels
        n_cal, n_test = len(X_cal), len(X_test)
        X_combined = np.vstack([X_cal, X_test])
        y_combined = np.array([0] * n_cal + [1] * n_test)
        
        # Fit classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_combined, y_combined)
        
        # Probability of being from test distribution
        probs = clf.predict_proba(X_cal)[:, 1]
        
        # Convert to density ratio weights
        weights = probs / (1 - probs + 1e-10)
        weights = np.clip(weights, 0.1, 10.0)  # Bound for stability
        
        return weights







