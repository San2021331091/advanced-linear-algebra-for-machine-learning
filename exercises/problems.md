# Exercises

Try these before looking up solutions. Use NumPy only after deriving the mathematics.

## Foundations
1. Find a basis and dimension for the span of (1, 2, 0), (2, 4, 0), and (0, 1, 1).
2. For a 4x6 matrix of rank 3, determine the nullity.
3. Explain why a linear map from R^5 to R^3 cannot be one-to-one.

## Orthogonality and factorizations
4. Project b=(2, 1, 3) onto the span of a=(1, 0, 1).
5. Implement modified Gram-Schmidt without calling `numpy.linalg.qr`.
6. Explain why QR is often preferable to normal equations for least squares.

## Spectral methods and SVD
7. Show that eigenvectors of a real symmetric matrix associated with distinct eigenvalues are orthogonal.
8. Prove that truncating the SVD gives the best rank-k approximation in Frobenius norm (state the Eckart-Young theorem).
9. Explain how singular values reveal effective rank.

## AI applications
10. Implement PCA and compare it with `sklearn.decomposition.PCA`.
11. Use rank-k SVD to compress a user-provided matrix; plot reconstruction error against k.
12. Derive the dimensions of Q, K, V and the attention score matrix for a sequence of length n.
13. Explain why scaled dot-product attention divides by sqrt(d_k).
14. Investigate how feature scaling changes PCA.

## Numerical computing
15. Compare `solve(A, b)` with `inv(A) @ b` on an ill-conditioned matrix.
16. Explain condition number in terms of sensitivity to perturbations.
17. Implement power iteration and compare its dominant eigenvalue estimate with `numpy.linalg.eigvalsh`.
