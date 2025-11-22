from scipy.io import mmread
from scipy.sparse import csr_matrix
import time
import psutil
import os

A = mmread('mc2depi/mc2depi.mtx')
A_csr = csr_matrix(A)

print(f"Matrix: mc2depi")
print(f"Size: {A_csr.shape[0]} × {A_csr.shape[1]}")
print(f"Non-zeros: {A_csr.nnz:,}")
print(f"Sparsity: {(1 - A_csr.nnz / (A_csr.shape[0] * A_csr.shape[1])) * 100:.4f}%")

mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

start = time.perf_counter()
result = A_csr @ A_csr.T
end = time.perf_counter()

mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

print(f"\nCSR Multiplication (A × A^T):")
print(f"  Time: {end - start:.4f}s")
print(f"  Memory: {mem_after:.2f} MB")
print(f"\nDense equivalent would need: {(A_csr.shape[0]**2 * 8) / (1024**3):.2f} GB")