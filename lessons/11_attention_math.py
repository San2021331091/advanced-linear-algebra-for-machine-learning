import numpy as np
from linear_algebra_ai.attention import scaled_dot_product_attention

rng = np.random.default_rng(0)
sequence_length, d_k, d_v = 4, 8, 6
Q = rng.normal(size=(sequence_length, d_k))
K = rng.normal(size=(sequence_length, d_k))
V = rng.normal(size=(sequence_length, d_v))
output, weights = scaled_dot_product_attention(Q, K, V)
print("attention weights (each row sums to 1):\n", weights)
print("row sums:", weights.sum(axis=1))
print("output shape:", output.shape)
