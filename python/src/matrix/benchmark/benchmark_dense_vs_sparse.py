import sys
import time
import csv
import os
import psutil
from python.src.matrix.dense.matrix import DenseMatrix
from python.src.matrix.dense.matrix_numpy import DenseMatrixNumPy
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR
from python.src.matrix.sparse.matrix_scipy import SparseMatrixSciPy


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_single(name, multiply_func, A, B):
    mem_before = get_process_memory_mb()
    
    start = time.perf_counter()
    result = multiply_func(A, B)
    end = time.perf_counter()
    
    mem_after = get_process_memory_mb()
    
    return (end - start, max(mem_before, mem_after))


def run_benchmark(size, sparsity, runs, writer):    
    print(f"Size {size}×{size}, Sparsity {sparsity*100:.0f}%")
    
    results = {'Dense-Python': [], 'Sparse-CSR': [], 'Dense-NumPy': [], 'Sparse-SciPy': []}
    nnz_list = []
    
    for run in range(runs):
        A_sparse_csr = SparseMatrixCSR.random(size, sparsity)
        B_sparse_csr = SparseMatrixCSR.random(size, sparsity)
        
        A_sparse_scipy = SparseMatrixSciPy.random(size, sparsity)
        B_sparse_scipy = SparseMatrixSciPy.random(size, sparsity)
        
        nnz_list.append(A_sparse_csr.numbers_non_zero())

        A_dense_python = DenseMatrix(A_sparse_csr.to_dense())
        B_dense_python = DenseMatrix(B_sparse_csr.to_dense())
        
        A_dense_numpy = DenseMatrixNumPy(A_sparse_scipy.to_dense())
        B_dense_numpy = DenseMatrixNumPy(B_sparse_scipy.to_dense())

        time_dp, mem_dp = benchmark_single("Dense-Python", lambda a, b: a.multiply_row_oriented(b), A_dense_python, B_dense_python)
        results['Dense-Python'].append((time_dp, mem_dp))
        
        time_sc, mem_sc = benchmark_single("Sparse-CSR", lambda a, b: a.multiply(b), A_sparse_csr, B_sparse_csr)
        results['Sparse-CSR'].append((time_sc, mem_sc))

        time_dn, mem_dn = benchmark_single("Dense-NumPy", lambda a, b: a.multiply_matmul(b), A_dense_numpy, B_dense_numpy)
        results['Dense-NumPy'].append((time_dn, mem_dn))
        
        time_ss, mem_ss = benchmark_single("Sparse-SciPy", lambda a, b: a.multiply(b), A_sparse_scipy, B_sparse_scipy)
        results['Sparse-SciPy'].append((time_ss, mem_ss))
    
    avg_nnz = sum(nnz_list) / len(nnz_list)
    actual_sparsity = 1 - (avg_nnz / (size * size))
    
    for algo_name, measurements in results.items():
        avg_time = sum(t for t, m in measurements) / len(measurements)
        avg_mem = sum(m for t, m in measurements) / len(measurements)
        
        writer.writerow([size, sparsity, round(actual_sparsity, 4), int(avg_nnz), algo_name, round(avg_time, 6), round(avg_mem, 2)])
    
    avg_time_dp = sum(t for t, m in results['Dense-Python']) / runs
    avg_time_sc = sum(t for t, m in results['Sparse-CSR']) / runs
    avg_time_dn = sum(t for t, m in results['Dense-NumPy']) / runs
    avg_time_ss = sum(t for t, m in results['Sparse-SciPy']) / runs
    
    speedup_csr = avg_time_dp / avg_time_sc if avg_time_sc > 0 else 0
    speedup_scipy = avg_time_dn / avg_time_ss if avg_time_ss > 0 else 0
    
    print(f"  Dense-Python: {avg_time_dp:.4f}s | Sparse-CSR: {avg_time_sc:.4f}s | Speedup: {speedup_csr:.2f}x")
    print(f"  Dense-NumPy: {avg_time_dn:.4f}s | Sparse-SciPy: {avg_time_ss:.4f}s | Speedup: {speedup_scipy:.2f}x\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_dense_vs_sparse.py <output_directory>")
        print("Example: python benchmark_dense_vs_sparse.py output/")
        sys.exit(1)
    
    sizes = [128, 256, 512]
    sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
    runs = 3
    
    output_directory = sys.argv[1]
    os.makedirs(output_directory, exist_ok=True)
    
    csv_path = os.path.join(output_directory, "dense_vs_sparse.csv")
    
    print("\nDENSE vs SPARSE COMPARISON")
    print(f"Sizes: {sizes}")
    print(f"Sparsity levels: {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"Runs per config: {runs}")
    print(f"Output: {csv_path}\n")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Size", "Sparsity", "ActualSparsity", "NonZeroElements", "Algorithm", "AvgTimeSeconds", "AvgMemoryMB"])
        
        for sparsity in sparsities:
            print(f"Sparsity {sparsity*100:.0f}%:")
            for size in sizes:
                run_benchmark(size, sparsity, runs, writer)
    
    print(f"Results saved: {csv_path}")