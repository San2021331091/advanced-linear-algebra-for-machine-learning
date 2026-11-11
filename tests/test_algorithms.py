import numpy as np
from linear_algebra_ai.core import gram_schmidt, rank_nullity
from linear_algebra_ai.decompositions import truncated_svd
from linear_algebra_ai.attention import scaled_dot_product_attention

def test_rank_nullity():
    A = np.array([[1., 2., 3.], [2., 4., 6.]])
    assert rank_nullity(A) == (1, 2)

def test_gram_schmidt_is_orthonormal():
    A = np.array([[1., 1.], [1., 0.], [0., 1.]])
    Q = gram_schmidt(A)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))

def test_truncated_svd_rank():
    A = np.arange(20, dtype=float).reshape(4, 5)
    Ak = truncated_svd(A, 1)
    assert np.linalg.matrix_rank(Ak, tol=1e-9) <= 1

def test_attention_weights_sum_to_one():
    Q = K = np.eye(2)
    V = np.array([[1., 0.], [0., 1.]])
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(weights.sum(axis=1), 1)
    assert output.shape == (2, 2)
