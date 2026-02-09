"""Distributionally Robust Optimization for portfolio construction.

Implements Wasserstein DRO which optimizes for the worst-case distribution
within a Wasserstein ball around the empirical distribution. This provides
robustness against distribution shift and model misspecification.

References:
- Esfahani & Kuhn, "Data-driven Distributionally Robust Optimization" (Math Programming 2018)
- Blanchet et al., "Quantifying Distributional Model Risk" (Math of OR 2019)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import cvxpy as cp
from typing import Optional, Tuple, Literal
import logging

logger = logging.getLogger(__name__)


class WassersteinDROOptimizer:
    """Wasserstein Distributionally Robust Portfolio Optimization.
    
    Instead of optimizing expected return, we optimize worst-case
    expected return over all distributions within Wasserstein distance
    ε of the empirical distribution:
    
        max_w min_{P: W(P, P̂) ≤ ε} E_P[w'r]
    
    This is equivalent to a robust mean-variance problem with
    uncertainty sets derived from the Wasserstein constraint.
    
    Attributes:
        radius: Wasserstein ball radius (distributional uncertainty)
        risk_aversion: Risk aversion parameter γ
        solver: Convex optimization solver
    """
    
    def __init__(
        self,
        radius: float = 0.05,
        risk_aversion: float = 1.0,
        solver: str = "CLARABEL",
        max_weight: float = 0.20,
        min_weight: float = 0.0,
        allow_short: bool = False
    ):
        self.radius = radius
        self.risk_aversion = risk_aversion
        self.solver = solver
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        
        logger.info(
            f"Initialized WassersteinDROOptimizer: radius={radius}, "
            f"risk_aversion={risk_aversion}"
        )
    
    def optimize(
        self,
        returns: NDArray,
        cov_matrix: Optional[NDArray] = None
    ) -> Tuple[NDArray, dict]:
        """Optimize portfolio using Wasserstein DRO.
        
        The worst-case expected return under Wasserstein uncertainty is:
            E[w'r] - ε * ||Σ^{1/2} w||_2
        
        This admits a tractable second-order cone program (SOCP).
        
        Args:
            returns: Historical returns of shape (n_samples, n_assets)
            cov_matrix: Covariance matrix (optional, computed if None)
            
        Returns:
            Tuple of (optimal_weights, result_dict)
        """
        n_samples, n_assets = returns.shape
        
        # Estimate moments
        mu = returns.mean(axis=0)
        if cov_matrix is None:
            cov_matrix = np.cov(returns, rowvar=False)
        
        # Cholesky decomposition for SOCP
        try:
            Sigma_sqrt = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add regularization for numerical stability
            cov_reg = cov_matrix + 1e-6 * np.eye(n_assets)
            Sigma_sqrt = np.linalg.cholesky(cov_reg)
        
        # Decision variables
        w = cp.Variable(n_assets)
        t = cp.Variable()  # Auxiliary for SOCP
        
        # Worst-case mean: μ - ε * ||Σ^{1/2} w||_2
        # We maximize: μ'w - ε * ||Σ^{1/2} w||_2 - γ/2 * w'Σw
        
        # Objective: max μ'w - ε*t - γ/2 * quad_form(w, Σ)
        objective = mu @ w - self.radius * t
        if self.risk_aversion > 0:
            objective -= (self.risk_aversion / 2) * cp.quad_form(w, cov_matrix)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Fully invested
            w >= self.min_weight,  # Min weight
            w <= self.max_weight,  # Max weight
            cp.SOC(t, Sigma_sqrt.T @ w)  # ||Σ^{1/2} w|| <= t
        ]
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=getattr(cp, self.solver))
        except Exception as e:
            logger.warning(f"Primary solver failed: {e}, trying ECOS")
            problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Optimization status: {problem.status}")
        
        weights = w.value
        
        # Compute worst-case metrics
        worst_case_mean = float(mu @ weights - self.radius * t.value)
        portfolio_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        
        result = {
            "weights": weights,
            "expected_return": float(mu @ weights),
            "worst_case_return": worst_case_mean,
            "volatility": portfolio_vol,
            "sharpe_ratio": worst_case_mean / portfolio_vol if portfolio_vol > 0 else 0,
            "robustness_cost": float(self.radius * t.value),
            "status": problem.status
        }
        
        logger.info(
            f"DRO optimization complete: worst_case_return={worst_case_mean:.4f}, "
            f"vol={portfolio_vol:.4f}"
        )
        
        return weights, result
    
    def optimize_with_conformal_bounds(
        self,
        returns: NDArray,
        return_bounds: Tuple[NDArray, NDArray],
        cov_matrix: Optional[NDArray] = None
    ) -> Tuple[NDArray, dict]:
        """Optimize with conformal prediction intervals as constraints.
        
        Uses conformal bounds on expected returns as additional
        uncertainty sets, combining DRO with conformal inference.
        
        Args:
            returns: Historical returns
            return_bounds: Tuple of (lower_bounds, upper_bounds) per asset
            cov_matrix: Covariance matrix
            
        Returns:
            Tuple of (optimal_weights, result_dict)
        """
        n_samples, n_assets = returns.shape
        mu = returns.mean(axis=0)
        
        if cov_matrix is None:
            cov_matrix = np.cov(returns, rowvar=False)
        
        lower_bounds, upper_bounds = return_bounds
        
        # Use worst-case from conformal bounds
        mu_robust = lower_bounds  # Conservative: use lower bound
        
        Sigma_sqrt = np.linalg.cholesky(
            cov_matrix + 1e-6 * np.eye(n_assets)
        )
        
        w = cp.Variable(n_assets)
        t = cp.Variable()
        
        # Robust objective with conformal lower bound
        objective = mu_robust @ w - self.radius * t
        if self.risk_aversion > 0:
            objective -= (self.risk_aversion / 2) * cp.quad_form(w, cov_matrix)
        
        constraints = [
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
            cp.SOC(t, Sigma_sqrt.T @ w)
        ]
        
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=getattr(cp, self.solver, cp.ECOS))
        
        weights = w.value
        
        result = {
            "weights": weights,
            "expected_return": float(mu @ weights),
            "robust_expected_return": float(mu_robust @ weights),
            "volatility": float(np.sqrt(weights @ cov_matrix @ weights)),
            "status": problem.status
        }
        
        return weights, result


class RobustPortfolioOptimizer:
    """Unified interface for robust portfolio optimization methods.
    
    Supports multiple robustness paradigms:
    - Wasserstein DRO
    - CVaR-based robust optimization
    - Factor uncertainty sets
    """
    
    def __init__(
        self,
        method: Literal["wasserstein_dro", "cvar_robust", "factor_robust"] = "wasserstein_dro",
        **kwargs
    ):
        self.method = method
        self.kwargs = kwargs
        
        if method == "wasserstein_dro":
            self.optimizer = WassersteinDROOptimizer(**kwargs)
        elif method == "cvar_robust":
            self.optimizer = CVaRRobustOptimizer(**kwargs)
        elif method == "factor_robust":
            self.optimizer = FactorRobustOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize(
        self,
        returns: NDArray,
        **kwargs
    ) -> Tuple[NDArray, dict]:
        """Optimize portfolio.
        
        Args:
            returns: Historical returns
            **kwargs: Additional method-specific arguments
            
        Returns:
            Tuple of (weights, result_dict)
        """
        return self.optimizer.optimize(returns, **kwargs)


class CVaRRobustOptimizer:
    """CVaR-based robust portfolio optimization.
    
    Minimizes portfolio CVaR while maintaining minimum expected return,
    providing robustness to tail risk.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        target_return: Optional[float] = None,
        max_weight: float = 0.20
    ):
        self.alpha = alpha
        self.target_return = target_return
        self.max_weight = max_weight
    
    def optimize(
        self,
        returns: NDArray,
        **kwargs
    ) -> Tuple[NDArray, dict]:
        """Optimize portfolio minimizing CVaR.
        
        Uses the Rockafellar-Uryasev LP formulation.
        """
        n_samples, n_assets = returns.shape
        mu = returns.mean(axis=0)
        
        w = cp.Variable(n_assets)
        var = cp.Variable()  # VaR threshold
        u = cp.Variable(n_samples)  # Auxiliary for CVaR
        
        # CVaR = VaR + (1/alpha) * E[max(loss - VaR, 0)]
        # Portfolio loss = -w'r
        portfolio_returns = returns @ w
        
        # Objective: minimize CVaR
        cvar = var + (1 / (self.alpha * n_samples)) * cp.sum(u)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight,
            u >= 0,
            u >= -portfolio_returns - var  # u_i >= loss_i - VaR
        ]
        
        if self.target_return is not None:
            constraints.append(mu @ w >= self.target_return)
        
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve(solver=cp.ECOS)
        
        weights = w.value
        
        result = {
            "weights": weights,
            "expected_return": float(mu @ weights),
            "cvar": float(cvar.value),
            "var": float(var.value),
            "status": problem.status
        }
        
        return weights, result


class FactorRobustOptimizer:
    """Factor-based robust optimization with uncertainty in loadings.
    
    Accounts for estimation error in factor exposures using
    ellipsoidal uncertainty sets.
    """
    
    def __init__(
        self,
        n_factors: int = 3,
        uncertainty_radius: float = 0.1,
        max_weight: float = 0.20
    ):
        self.n_factors = n_factors
        self.uncertainty_radius = uncertainty_radius
        self.max_weight = max_weight
    
    def optimize(
        self,
        returns: NDArray,
        factor_returns: Optional[NDArray] = None,
        **kwargs
    ) -> Tuple[NDArray, dict]:
        """Optimize with factor uncertainty.
        
        Uses PCA factors if not provided.
        """
        n_samples, n_assets = returns.shape
        
        # Extract factors via PCA if not provided
        if factor_returns is None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_factors)
            factor_returns = pca.fit_transform(returns)
            loadings = pca.components_.T
        else:
            # Estimate loadings via regression
            loadings = np.linalg.lstsq(
                factor_returns, returns, rcond=None
            )[0].T
        
        # Factor covariance
        factor_cov = np.cov(factor_returns, rowvar=False)
        residual_var = np.var(
            returns - factor_returns @ loadings.T, axis=0
        )
        
        mu = returns.mean(axis=0)
        
        w = cp.Variable(n_assets)
        
        # Robust objective accounting for loading uncertainty
        # Worst-case factor exposure: ||B'w|| * uncertainty_radius
        factor_exposure = loadings.T @ w
        
        # Robust variance = w'(B Σ_f B' + D)w + ε * ||B'w||
        systematic_var = cp.quad_form(factor_exposure, factor_cov)
        idio_var = residual_var @ cp.square(w)
        robust_var = systematic_var + idio_var + \
                     self.uncertainty_radius * cp.norm(factor_exposure)
        
        objective = mu @ w - 0.5 * robust_var
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight
        ]
        
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.ECOS)
        
        weights = w.value
        
        result = {
            "weights": weights,
            "expected_return": float(mu @ weights),
            "factor_exposure": loadings.T @ weights,
            "status": problem.status
        }
        
        return weights, result







