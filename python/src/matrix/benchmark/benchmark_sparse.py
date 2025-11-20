import sys
import time
import csv
import os
import psutil
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR
from python.src.matrix.sparse.matrix_scipy import SparseMatrixSciPy


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(algorithm_name, multiply_func, generate_func, sizes, sparsities, runs, writer):
    print(f"\nBenchmarking: {algorithm_name}")
    
    for sparsity in sparsities:
        print(f"  Sparsity {sparsity*100:.0f}%:")
        
        for size in sizes:
            print(f"    Size {size}×{size}...", end=' ')
            
            times = []
            memories = []
            nnz_list = []
            
            for run in range(1, runs + 1):
                A, B = generate_func(size, sparsity)
                
                mem_before = get_process_memory_mb()
                
                start = time.perf_counter()
                result = multiply_func(A, B)
                end = time.perf_counter()

                mem_after = get_process_memory_mb()
                
                time_seconds = round(end - start, 6)
                memory_mb = round(max(mem_before, mem_after), 2)
                nnz = A.numbers_non_zero()
                actual_sparsity = A.get_sparsity()
                
                times.append(time_seconds)
                memories.append(memory_mb)
                nnz_list.append(nnz)
                
                writer.writerow([algorithm_name, size, sparsity, run, time_seconds, memory_mb, nnz, actual_sparsity])
            
            avg_time = sum(times) / len(times)
            avg_memory = sum(memories) / len(memories)
            avg_nnz = sum(nnz_list) / len(nnz_list)
            print(f"Avg: {avg_time:.4f}s, {avg_memory:.2f}MB, NNZ: {avg_nnz:.0f}")


def run_all_benchmarks(sizes, sparsities, runs, csv_path):    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Algorithm", "Size", "Sparsity", "Run", "TimeSeconds", "MemoryMB", "NonZeroElements", "ActualSparsity"])
        
        print("\nPYTHON PURE SPARSE ALGORITHMS")
    
        run_benchmark("CSR-Pure", lambda A, B: A.multiply(B), lambda n, s: (SparseMatrixCSR.random(n, s), SparseMatrixCSR.random(n, s)), sizes, sparsities, runs, writer)
        
        print("\nSCIPY SPARSE ALGORITHMS")
        run_benchmark("CSR-SciPy", lambda A, B: A.multiply(B), lambda n, s: (SparseMatrixSciPy.random(n, s), SparseMatrixSciPy.random(n, s)), sizes, sparsities, runs, writer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_sparse.py <output_directory>")
        print("Example: python benchmark_sparse.py results/")
        sys.exit(1)
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
    runs = 3
    
    output_directory = sys.argv[1]
    os.makedirs(output_directory, exist_ok=True)
    
    csv_path = os.path.join(output_directory, "sparse_algorithms.csv")
    
    print("SPARSE MATRIX MULTIPLICATION BENCHMARK")
    print(f"\nConfiguration:")
    print(f"  Sizes: {sizes}")
    print(f"  Sparsity levels: {[f'{s*100:.0f}%' for s in sparsities]}")
    print(f"  Runs per configuration: {runs}")
    print(f"  Output: {csv_path}")

    run_all_benchmarks(sizes, sparsities, runs, csv_path)

    print(f"\nResults saved at: {csv_path}")
