import numpy as np

def rank_nullity(A, tol=None):
    """Return (rank, nullity), using rank-nullity: rank(A)+nullity(A)=n."""
    A = np.asarray(A, dtype=float)
    rank = int(np.linalg.matrix_rank(A, tol=tol))
    return rank, A.shape[1] - rank

def project_onto_columns(A, b):
    """Project b onto Col(A), returning projection and least-squares coefficients."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return A @ coeffs, coeffs

def gram_schmidt(A, tol=1e-12):
    """Orthonormalize the columns of A (modified Gram-Schmidt)."""
    A = np.asarray(A, dtype=float)
    Q = np.zeros_like(A)
    count = 0
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(count):
            v -= np.dot(Q[:, i], v) * Q[:, i]
        norm = np.linalg.norm(v)
        if norm > tol:
            Q[:, count] = v / norm
            count += 1
    return Q[:, :count]
