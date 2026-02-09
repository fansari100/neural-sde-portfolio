# Neural SDE Portfolio Optimizer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**A mathematically rigorous portfolio optimization framework combining Neural Stochastic Differential Equations (Neural SDEs), Path Signature methods from Rough Path Theory, and Conformal Prediction for distribution-free uncertainty quantification.**

## 🔬 What Makes This Innovative

Unlike traditional ML trading systems that treat price series as discrete sequences, this framework:

1. **Neural SDEs**: Models asset dynamics as continuous-time stochastic processes, learning the drift and diffusion functions via neural networks
2. **Path Signatures**: Extracts features using rough path theory - a mathematically grounded approach that captures sequential patterns without arbitrary windowing
3. **Conformal Prediction**: Provides distribution-free, finite-sample valid prediction intervals without assuming Gaussianity
4. **Controlled Differential Equations**: Uses Neural CDEs for irregularly-sampled financial data

## 🧮 Mathematical Foundation

### Neural Stochastic Differential Equations

Standard SDE:
```
dX_t = μ(X_t, t)dt + σ(X_t, t)dW_t
```

Neural SDE (learned dynamics):
```
dX_t = f_θ(X_t, t)dt + g_φ(X_t, t)dW_t
```

Where `f_θ` and `g_φ` are neural networks learning drift and diffusion.

### Path Signatures (Rough Path Theory)

The signature of a path `X: [0,T] → ℝᵈ` is the sequence of iterated integrals:
```
S(X)_{[s,t]} = (1, ∫ dX, ∫∫ dX⊗dX, ∫∫∫ dX⊗dX⊗dX, ...)
```

**Key Property**: The signature uniquely characterizes the path up to tree-like equivalence and is invariant to time reparametrization.

### Conformal Prediction

Given calibration data, conformal prediction constructs prediction sets `C(X_{n+1})` such that:
```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
```

This holds **without any distributional assumptions** - critical for heavy-tailed financial returns.

## 🏗️ Architecture

```
neural-sde-portfolio/
├── src/
│   ├── sde/
│   │   ├── neural_sde.py        # Neural SDE implementation (torchsde)
│   │   ├── diffusion.py         # Learned diffusion functions
│   │   ├── drift.py             # Learned drift functions
│   │   └── solver.py            # SDE solvers (Euler-Maruyama, Milstein)
│   ├── signatures/
│   │   ├── signature.py         # Path signature computation (signatory)
│   │   ├── logsig.py            # Log-signature features
│   │   ├── augmentations.py     # Time/lead-lag augmentations
│   │   └── kernel.py            # Signature kernel for similarity
│   ├── conformal/
│   │   ├── cqr.py               # Conformalized Quantile Regression
│   │   ├── aci.py               # Adaptive Conformal Inference
│   │   └── online.py            # Online conformal prediction
│   ├── cde/
│   │   ├── neural_cde.py        # Neural Controlled Differential Equations
│   │   └── ncde_model.py        # NCDE for irregular time series
│   ├── portfolio/
│   │   ├── optimizer.py         # Mean-variance with conformal bounds
│   │   ├── robust.py            # Distributionally robust optimization
│   │   └── continuous.py        # Continuous-time rebalancing
│   └── utils/
│       ├── preprocessing.py     # Data preprocessing
│       └── metrics.py           # Performance metrics
├── experiments/
│   ├── benchmark_signatures.py
│   ├── neural_sde_training.py
│   └── conformal_coverage.py
├── tests/
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

```python
import torch
from src.sde import NeuralSDE, SDESolver
from src.signatures import SignatureFeatures
from src.conformal import ConformalizedQuantileRegression
from src.portfolio import RobustPortfolioOptimizer

# 1. Extract path signature features from price data
sig_extractor = SignatureFeatures(
    depth=4,
    augmentations=["time", "leadlag", "basepoint"]
)
features = sig_extractor.transform(price_paths)  # Shape: (batch, sig_dim)

# 2. Train Neural SDE for continuous-time dynamics
neural_sde = NeuralSDE(
    drift_net=DriftNetwork(hidden_dims=[64, 64]),
    diffusion_net=DiffusionNetwork(hidden_dims=[64, 64]),
    noise_type="diagonal"
)
solver = SDESolver(method="milstein", dt=1/252)
trajectories = solver.sample(neural_sde, initial_state, n_paths=10000)

# 3. Conformal prediction for uncertainty quantification
cqr = ConformalizedQuantileRegression(
    model=QuantileRegressor(),
    alpha=0.10  # 90% coverage
)
cqr.calibrate(X_cal, y_cal)
lower, upper = cqr.predict(X_test)  # Valid prediction intervals

# 4. Robust portfolio optimization with uncertainty
optimizer = RobustPortfolioOptimizer(
    method="wasserstein_dro",
    radius=0.05
)
weights = optimizer.optimize(
    expected_returns=mu,
    covariance=Sigma,
    uncertainty_sets=(lower, upper)
)
```

## 📊 Key Innovations

### 1. Signature-Augmented Neural SDE

```python
class SignatureAugmentedSDE(nn.Module):
    """SDE with signature-informed drift and diffusion."""
    
    def __init__(self, sig_depth: int = 4):
        self.sig_layer = SignatureLayer(depth=sig_depth)
        self.drift = nn.Sequential(
            nn.Linear(sig_dim + state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, state_dim)
        )
        self.diffusion = nn.Sequential(
            nn.Linear(sig_dim + state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, state_dim),
            nn.Softplus()  # Ensure positive diffusion
        )
    
    def f(self, t, y, path_history):
        """Drift function μ(t, y, history)."""
        sig = self.sig_layer(path_history)
        return self.drift(torch.cat([y, sig], dim=-1))
    
    def g(self, t, y, path_history):
        """Diffusion function σ(t, y, history)."""
        sig = self.sig_layer(path_history)
        return self.diffusion(torch.cat([y, sig], dim=-1))
```

### 2. Adaptive Conformal Inference for Non-Exchangeable Data

Standard conformal prediction assumes exchangeability, which fails for time series. We implement **Adaptive Conformal Inference (ACI)** which maintains coverage under distribution shift:

```python
class AdaptiveConformalInference:
    """ACI for time series with coverage guarantees under shift."""
    
    def __init__(self, alpha: float, gamma: float = 0.01):
        self.alpha = alpha
        self.gamma = gamma  # Learning rate for adaptation
        self.alpha_t = alpha  # Adaptive miscoverage rate
    
    def update(self, y_true, interval):
        """Update alpha_t based on observed coverage."""
        covered = interval[0] <= y_true <= interval[1]
        # Gradient descent on miscoverage
        self.alpha_t += self.gamma * (self.alpha - (1 - covered))
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.50)
```

### 3. Wasserstein Distributionally Robust Optimization

Instead of point estimates, optimize over worst-case distributions within a Wasserstein ball:

```python
def wasserstein_dro_portfolio(mu, Sigma, radius: float):
    """
    min_w max_{P: W(P, P_hat) <= radius} E_P[-w'r]
    
    Equivalent to robust mean-variance with uncertainty set.
    """
    n = len(mu)
    
    # Worst-case mean: mu - radius * Sigma^{1/2} * w / ||w||
    # Solved via second-order cone programming
    w = cp.Variable(n)
    kappa = cp.Variable()  # Auxiliary for SOCP
    
    objective = mu @ w - radius * kappa
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.SOC(kappa, Sigma_sqrt @ w)  # ||Σ^{1/2} w|| <= kappa
    ]
    
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.MOSEK)
    
    return w.value
```

## 📈 Theoretical Guarantees

| Property | Guarantee |
|----------|-----------|
| Signature universality | Signatures uniquely identify paths (Chen's theorem) |
| SDE solution existence | Lipschitz drift/diffusion ensure strong solutions |
| Conformal coverage | P(Y ∈ C(X)) ≥ 1-α for any distribution |
| DRO robustness | Worst-case performance over uncertainty set |

## 🔧 Dependencies

```
torch>=2.0.0
torchsde>=0.2.5          # Neural SDE library
signatory>=1.2.6         # Path signatures
torchcde>=0.2.5          # Neural CDEs
cvxpy>=1.4.0             # Convex optimization
mosek>=10.0              # SOCP solver
scipy>=1.11.0
numpy>=1.24.0
```

## 📚 References

1. **Neural SDEs**: Li et al., "Scalable Gradients for Stochastic Differential Equations" (AISTATS 2020)
2. **Path Signatures**: Kidger & Lyons, "Signatory: Differentiable Computations of the Signature" (ICLR 2021)
3. **Conformal Prediction**: Romano et al., "Conformalized Quantile Regression" (NeurIPS 2019)
4. **Adaptive CI**: Gibbs & Candès, "Adaptive Conformal Inference Under Distribution Shift" (NeurIPS 2021)
5. **Wasserstein DRO**: Esfahani & Kuhn, "Data-driven Distributionally Robust Optimization" (Math Programming 2018)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**Ricky Ansari**  
[GitHub](https://github.com/fansari100) | [LinkedIn](https://linkedin.com/in/ricky-ansari-053143133) | [Website](https://rickyansari.com)







