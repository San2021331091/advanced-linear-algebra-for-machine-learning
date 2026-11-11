import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Single-head scaled dot-product attention.
    Q: (query_length, d_k), K: (key_length, d_k), V: (key_length, d_v).
    """
    Q, K, V = map(lambda x: np.asarray(x, dtype=float), (Q, K, V))
    scores = (Q @ K.T) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
