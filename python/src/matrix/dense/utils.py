import random
import numpy as np

def generate_matrices(n):
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    return A, B

def generate_matrices_numpy(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return A, B