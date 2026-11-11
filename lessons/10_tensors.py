import numpy as np

# A batch of 2 matrices, each 3x4: a rank-3 tensor.
T = np.arange(24).reshape(2, 3, 4)
print("tensor shape:", T.shape)
print("slice for batch item 0:\n", T[0])
# Matrix multiplication independently for every batch item:
A = np.ones((2, 3, 5))
B = np.ones((2, 5, 4))
C = A @ B
print("batched matrix product shape:", C.shape)
