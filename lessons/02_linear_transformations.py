import numpy as np

# A linear layer without bias is the map x -> Wx.
W = np.array([[1., 2., 0.], [0., 1., 1.]])
x = np.array([2., -1., 3.])
print("input shape:", x.shape)
print("output:", W @ x)
print("rank and nullity:", np.linalg.matrix_rank(W), W.shape[1] - np.linalg.matrix_rank(W))
