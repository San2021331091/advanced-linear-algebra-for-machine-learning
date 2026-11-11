import numpy as np
from linear_algebra_ai.core import gram_schmidt, project_onto_columns

A = np.array([[1., 1.], [1., 0.], [0., 1.]])
Q = gram_schmidt(A)
print("Q.T @ Q (should be identity):\n", Q.T @ Q)
b = np.array([1., 2., 3.])
projection, coefficients = project_onto_columns(A, b)
print("projection:", projection)
print("residual orthogonal to Col(A):", A.T @ (b - projection))
