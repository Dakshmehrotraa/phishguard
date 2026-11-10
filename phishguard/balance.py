"""Class balancing with SMOTE.

Uses imbalanced-learn when installed; otherwise a small built-in SMOTE
(same algorithm: interpolate between a minority sample and one of its
k nearest minority neighbours). Works on dense arrays and sparse matrices.

Apply ONLY to the training split - never to validation or test data.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _builtin_smote(X, y, k_neighbors=5, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts.min() == counts.max():
        return X, y
    minority = classes[np.argmin(counts)]
    n_new = counts.max() - counts.min()
    X_min = X[y == minority]
    k = min(k_neighbors, X_min.shape[0] - 1)
    if k < 1:
        return X, y
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neigh = nn.kneighbors(X_min, return_distance=False)[:, 1:]
    base = rng.integers(0, X_min.shape[0], n_new)
    partner = neigh[base, rng.integers(0, k, n_new)]
    gap = rng.random(n_new)
    if sp.issparse(X_min):
        A, B = X_min[base], X_min[partner]
        X_new = A + sp.diags(gap) @ (B - A)
        X_out = sp.vstack([X, X_new]).tocsr()
    else:
        A, B = X_min[base], X_min[partner]
        X_new = A + gap[:, None] * (B - A)
        X_out = np.vstack([X, X_new])
    return X_out, np.concatenate([y, np.full(n_new, minority)])


def smote(X, y, random_state=42):
    """Return (X_resampled, y_resampled, method_used)."""
    try:
        from imblearn.over_sampling import SMOTE
        Xr, yr = SMOTE(random_state=random_state).fit_resample(X, y)
        return Xr, yr, "imblearn.SMOTE"
    except ImportError:
        Xr, yr = _builtin_smote(X, y, random_state=random_state)
        return Xr, yr, "built-in SMOTE (imbalanced-learn not installed)"
