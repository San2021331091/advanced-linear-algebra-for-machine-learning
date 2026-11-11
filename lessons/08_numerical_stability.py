import numpy as np
from linear_algebra_ai.decompositions import condition_number

# Hilbert matrices become increasingly ill-conditioned.
def hilbert(n):
    return np.fromfunction(lambda i, j: 1.0 / (i + j + 1), (n, n))

for n in (3, 5, 8):
    H = hilbert(n)
    print(f"n={n}, condition number={condition_number(H):.3e}")
