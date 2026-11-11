import numpy as np
from linear_algebra_ai.decompositions import truncated_svd

# Low-rank approximation is useful for compression and recommendation systems.
A = np.array([[5., 4., 3.], [4., 4., 2.], [1., 2., 0.], [0., 1., 1.]])
for k in (1, 2):
    Ak = truncated_svd(A, k)
    print(f"rank-{k} approximation error:", np.linalg.norm(A - Ak, ord="fro"))
