import numpy as np

# Symmetric matrices have real eigenvalues and orthogonal eigenspaces.
C = np.array([[4., 1.], [1., 3.]])
values, vectors = np.linalg.eigh(C)
print("eigenvalues:", values)
print("eigenvectors:\n", vectors)
print("check C V = V Lambda:", np.allclose(C @ vectors, vectors @ np.diag(values)))
