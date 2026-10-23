"""Feunou--Fontaine (2018), section 2.2, equations (3)--(4).

This is the paper's recursive predictive factor, not a fitted no-arbitrage DTSM.
Its decay and forward-rate loadings are jointly estimated from mature labels.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

PUBLISHED_ALPHA = {1: .8320, 2: .8298, 3: .8170, 6: .7749, 9: .7096, 12: .6284}


def recursive_factor(forwards, gamma, alpha, initial):
    """R[t]=(1-alpha)*gamma'[1,f[t]]+alpha*R[t-1], retaining every month."""
    f = np.asarray(forwards, float)
    if f.ndim != 2 or not np.isfinite(f).all() or not 0 <= alpha < 1:
        raise ValueError("Finite calendar-complete forwards and decay in [0,1) required")
    design = np.column_stack([np.ones(len(f)), f])
    return lfilter([1-alpha], [1, -alpha], design@np.asarray(gamma), zi=[alpha*initial])[0]


class MS17Regression:
    def fit(self, forward_history, returns, train_positions=None, *, initial=None):
        """Fit joint nonlinear LS with mean maturity loading fixed to one.

        forward_history includes known yield observations in the embargo gap;
        returns includes mature training outcomes only. train_positions maps
        those outcomes into the complete monthly forward calendar.
        """
        f, y = np.asarray(forward_history, float), np.asarray(returns, float)
        positions = np.arange(len(y)) if train_positions is None else np.asarray(train_positions, int)
        if f.ndim != 2 or y.ndim != 2 or len(positions) != len(y) or len(y) < 24:
            raise ValueError("Invalid training dimensions")
        if positions.min() < 0 or positions.max() >= len(f) or not np.isfinite(y).all():
            raise ValueError("Unmapped or nonfinite training labels")
        self.location_ = f[positions].mean(0)
        self.scale_ = np.maximum(f[positions].std(0), 1e-8)
        z = (f-self.location_)/self.scale_
        X = np.column_stack([np.ones(len(z)), z])
        self.initial_ = float(y.mean() if initial is None else initial)
        k, m = X.shape[1], y.shape[1]
        def unpack(theta):
            return theta[0], theta[1:k+1], np.r_[theta[k+1:], m-theta[k+1:].sum()]
        def residual(theta):
            alpha, gamma, b = unpack(theta)
            r = lfilter([1-alpha], [1, -alpha], X@gamma, zi=[alpha*self.initial_])[0]
            return (r[positions, None]*b-y).ravel()
        fits = []
        for a in (0., .6, .85):
            D = lfilter([1-a], [1, -a], X, axis=0)
            offset = a**(np.arange(len(X))+1)*self.initial_
            g = np.linalg.lstsq(D[positions], y.mean(1)-offset[positions], rcond=None)[0]
            theta = np.r_[a, g, np.ones(m-1)]
            lower, upper = np.full(len(theta), -np.inf), np.full(len(theta), np.inf)
            lower[0], upper[0] = 0., .995
            fit = least_squares(residual, theta, bounds=(lower, upper), max_nfev=250,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            fits.append(fit)
        best = min(fits, key=lambda fit: np.sum(fit.fun**2))
        self.alpha_, normalized_gamma, self.loadings_ = unpack(best.x)
        self.gamma_ = np.r_[normalized_gamma[0]-np.sum(normalized_gamma[1:]*self.location_/self.scale_),
                           normalized_gamma[1:]/self.scale_]
        self.loss_ = float(np.sum(best.fun**2))
        self.converged_ = bool(best.success)
        if not np.isfinite(self.loss_):
            raise ArithmeticError("MS17 nonlinear fit failed")
        return self

    def factor(self, forward_history):
        return recursive_factor(forward_history, self.gamma_, self.alpha_, self.initial_)

    def predict(self, forward_history):
        return self.factor(forward_history)[:, None]*self.loadings_
