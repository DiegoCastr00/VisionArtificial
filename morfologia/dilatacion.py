import numpy as np
def dilatacion(A, B):
    m, n = A.shape
    p, q = B.shape
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    if B[k, l] == 1 and i+k-p//2 >= 0 and i+k-p//2 < m and j+l-q//2 >= 0 and j+l-q//2 < n:
                        C[i, j] = max(C[i, j], A[i+k-p//2, j+l-q//2])                 
    return C