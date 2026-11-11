import numpy as np

def pca(X, n_components):
    """
    Principal component analysis.
    Rows are observations; columns are features.
    Returns transformed data, principal directions, and explained-variance ratios.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    X_centered = X - mean
    _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    transformed = X_centered @ components.T
    variances = singular_values ** 2 / max(X.shape[0] - 1, 1)
    ratios = variances / variances.sum()
    return transformed, components, ratios[:n_components], mean
