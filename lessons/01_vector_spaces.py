import numpy as np

# The columns of A are vectors. The column space is their span.
A = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1]], dtype=float)
print("rank:", np.linalg.matrix_rank(A))
print("basis for column space (via SVD):")
U, s, Vt = np.linalg.svd(A, full_matrices=False)
rank = np.linalg.matrix_rank(A)
print(U[:, :rank])
print("Interpretation: the rank counts independent directions in the feature space.")
