import numpy as np

# Fit y ~= X beta. Solve directly; do not form inv(X.T @ X).
X = np.array([[1., 0.], [1., 1.], [1., 2.], [1., 3.]])
y = np.array([1.1, 2.0, 2.9, 4.2])
beta, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
print("parameters:", beta)
print("predictions:", X @ beta)
print("residual sum of squares:", np.sum((y - X @ beta) ** 2))
