# Advanced Linear Algebra for AI

A hands-on, from-first-principles learning repository for the linear algebra used in
machine learning, deep learning, computer vision, and modern language models.

The goal is not only to call NumPy functions. Each topic combines:
1. the mathematical idea,
2. a NumPy implementation,
3. an AI/ML application,
4. exercises to check understanding.

## Learning path

| Order | Module | Core ideas | AI connection |
|---|---|---|---|
| 01 | Vector spaces | span, basis, dimension, subspaces | feature spaces and representations |
| 02 | Linear transformations | kernel, image, rank-nullity | neural-network layers |
| 03 | Orthogonality | inner products, projections, Gram-Schmidt | least squares and embeddings |
| 04 | Matrix factorizations | LU, QR, Cholesky | efficient numerical computation |
| 05 | Eigenvalues | eigenspaces, diagonalization, spectral theorem | graph learning and stability |
| 06 | SVD | singular values, low-rank approximation | compression, recommendation, PCA |
| 07 | Least squares | normal equations, pseudoinverse | regression |
| 08 | Numerical linear algebra | conditioning, floating-point error, iterative methods | reliable model training |
| 09 | PCA | covariance, eigendecomposition, explained variance | dimensionality reduction |
| 10 | Tensors | tensor products, matricization, contractions | deep-learning data and weights |
| 11 | Attention mathematics | dot products, projections, softmax geometry | Transformer attention |

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Run a lesson:

```bash
python lessons/03_orthogonality.py
python lessons/06_svd.py
python lessons/09_pca.py
python lessons/11_attention_math.py
```

Run tests:

```bash
python -m pytest
```
 
 
## Run On Colab: (Click the link)[https://colab.research.google.com/drive/10pl9sCe-C3XD6MG4_-480GDMD9TiKOth?usp=sharing]