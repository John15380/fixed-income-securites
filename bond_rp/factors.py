"""Shared two-stage factor estimator used by paper checks and live-style forecasts."""
import numpy as np
from sklearn.linear_model import LinearRegression

def common_index(X, y, current, *, second_intercept=True):
    """OLS average-return index followed by maturity loadings.

    CP (2005), eq. (2), restricts the second stage to b_n * index: the
    first-stage index already contains a constant. LN Table 2 includes a
    separate intercept. This function is shared by replication and forecasts.
    """
    y = np.asarray(y, dtype=float)
    first = LinearRegression().fit(X, y.mean(axis=1))
    index = first.predict(X).reshape(-1, 1)
    now = first.predict(current).reshape(-1, 1)
    second = LinearRegression(fit_intercept=second_intercept).fit(index, y)
    prediction = second.predict(now)
    np.testing.assert_allclose(prediction.mean(axis=1), now.ravel(), atol=1e-7, rtol=1e-7)
    return prediction, first, second



def ln6_terms(scores):
    """LN (2009): F1, F1 cubed, F2, F3, F4, F8, from eight PCs."""
    scores = np.asarray(scores)
    return np.column_stack([scores[:, 0], scores[:, 0]**3, scores[:, 1:4], scores[:, 7]])
