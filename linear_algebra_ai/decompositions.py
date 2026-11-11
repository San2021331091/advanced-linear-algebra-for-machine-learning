import numpy as np

def truncated_svd(A, k):
    """Best rank-k approximation in Frobenius norm (Eckart-Young theorem)."""
    U, s, Vt = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def qr_solve(A, b):
    """Solve a full-column-rank least-squares problem using QR."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    Q, R = np.linalg.qr(A, mode="reduced")
    return np.linalg.solve(R, Q.T @ b)

def condition_number(A, p=2):
    """Matrix condition number; a large value signals sensitivity."""
    return float(np.linalg.cond(np.asarray(A, dtype=float), p=p))
