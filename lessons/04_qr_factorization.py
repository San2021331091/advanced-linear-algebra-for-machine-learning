import numpy as np
from linear_algebra_ai.decompositions import qr_solve

A = np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
b = np.array([2.1, 2.9, 4.2, 4.8])
Q, R = np.linalg.qr(A, mode="reduced")
print("reconstruction error:", np.linalg.norm(A - Q @ R))
print("least-squares coefficients [intercept, slope]:", qr_solve(A, b))
