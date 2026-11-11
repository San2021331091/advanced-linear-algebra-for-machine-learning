import numpy as np
from linear_algebra_ai.dimensionality_reduction import pca

rng = np.random.default_rng(7)
latent = rng.normal(size=(300, 2))
mixing = np.array([[3., 1.], [1., 2.], [2., -1.], [0.5, 1.]])
X = latent @ mixing.T + 0.08 * rng.normal(size=(300, 4))
Z, components, ratios, mean = pca(X, n_components=2)
print("original shape:", X.shape)
print("compressed shape:", Z.shape)
print("explained variance ratios:", ratios)
print("total retained variance:", ratios.sum())
