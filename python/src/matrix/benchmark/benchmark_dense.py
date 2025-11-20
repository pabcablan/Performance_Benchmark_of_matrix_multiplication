import sys
import time
import csv
import os
import psutil
from python.src.matrix.dense.matrix import DenseMatrix
from python.src.matrix.dense.matrix_numpy import DenseMatrixNumPy


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(algorithm_name, multiply_func, generate_func, sizes, runs, writer):
    print(f"Benchmarking: {algorithm_name}")
    
    for size in sizes:
        print(f"Size {size}×{size}...", end=' ')
        
        times = []
        memories = []
        
        for run in range(1, runs + 1):
            A, B = generate_func(size)
            
            mem_before = get_process_memory_mb()
            
            start = time.perf_counter()
            result = multiply_func(A, B)
            end = time.perf_counter()

            mem_after = get_process_memory_mb()
            
            time_seconds = round(end - start, 6)
            memory_mb = round(max(mem_before, mem_after), 2)
            
            times.append(time_seconds)
            memories.append(memory_mb)
            
            writer.writerow([algorithm_name, size, run, time_seconds, memory_mb])
        
        avg_time = sum(times) / len(times)
        avg_memory = sum(memories) / len(memories)
        print(f"Avg: {avg_time:.4f}s, {avg_memory:.2f}MB")



def run_all_benchmarks(sizes, runs, csv_path):    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Algorithm", "Size", "Run", "TimeSeconds", "MemoryMB"])
        
        print("\nPYTHON PURE ALGORITHMS")
        
        run_benchmark("Standard", lambda A, B: A.multiply_standard(B), lambda n: (DenseMatrix.random(n), DenseMatrix.random(n)), sizes, runs, writer)
        run_benchmark("Row-Oriented", lambda A, B: A.multiply_row_oriented(B), lambda n: (DenseMatrix.random(n), DenseMatrix.random(n)), sizes, runs, writer)
        run_benchmark("Tiled-32", lambda A, B: A.multiply_tiled(B, 32), lambda n: (DenseMatrix.random(n), DenseMatrix.random(n)), sizes, runs, writer)
        run_benchmark("Tiled-64", lambda A, B: A.multiply_tiled(B, 64), lambda n: (DenseMatrix.random(n), DenseMatrix.random(n)), sizes, runs, writer)
        run_benchmark("Strassen", lambda A, B: A.multiply_strassen(B), lambda n: (DenseMatrix.random(n), DenseMatrix.random(n)), sizes, runs, writer)
        
        print("\nNUMPY ALGORITHMS")

        run_benchmark("NumPy-builtin", lambda A, B: A.multiply_builtin(B), lambda n: (DenseMatrixNumPy.random(n), DenseMatrixNumPy.random(n)), sizes, runs, writer)
        run_benchmark("NumPy-matmul", lambda A, B: A.multiply_matmul(B), lambda n: (DenseMatrixNumPy.random(n), DenseMatrixNumPy.random(n)), sizes, runs, writer)
        run_benchmark("NumPy-Tiled-64", lambda A, B: A.multiply_tiled(B, 64), lambda n: (DenseMatrixNumPy.random(n), DenseMatrixNumPy.random(n)), sizes, runs, writer)
        run_benchmark("NumPy-Strassen", lambda A, B: A.multiply_strassen(B), lambda n: (DenseMatrixNumPy.random(n), DenseMatrixNumPy.random(n)), sizes, runs, writer)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_dense.py <output_directory>")
        print("Example: python benchmark_dense.py results/")
        sys.exit(1)
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    runs = 1
    
    output_directory = sys.argv[1]
    os.makedirs(output_directory, exist_ok=True)
    
    csv_path = os.path.join(output_directory, "dense_algorithms.csv")
    
    print("DENSE MATRIX MULTIPLICATION BENCHMARK")
    print(f"\nConfiguration:")
    print(f"  Sizes: {sizes}")
    print(f"  Runs per size: {runs}")
    print(f"  Output: {csv_path}")

    run_all_benchmarks(sizes, runs, csv_path)

    print(f"Results saved at: {csv_path}")
