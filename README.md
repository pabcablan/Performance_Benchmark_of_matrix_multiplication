# Performance Benchmark of Matrix Multiplication: Dense vs Sparse Optimization Techniques

## Introduction

This repository contains a comprehensive benchmarking framework comparing dense and sparse matrix multiplication implementations across multiple optimization strategies. The study evaluates pure Python implementations (Standard, Row-Oriented, Tiled, Strassen) against library-optimized solutions (NumPy, SciPy) to determine performance crossover points and guide optimal representation selection based on sparsity levels.

**Key Findings:**
- 🚀 NumPy achieves **8,550× speedup** over pure Python for dense matrices
- 🎯 SciPy achieves **13,633× speedup** at 99% sparsity
- 📊 Sparse formats outperform dense at **70% sparsity** (pure Python) and **95% sparsity** (libraries)
- ⚡ Real-world validation: 525,825×525,825 matrix computed in **59ms** using **185 MB** (vs 2 TB dense)


## 🎯 Features

- **Dense Matrix Algorithms:**
  - Standard (ijk), Row-Oriented (ikj), Tiled (32, 64), Strassen
  - NumPy-optimized variants (builtin, matmul, Tiled, Strassen)

- **Sparse Matrix Algorithms:**
  - Compressed Sparse Row (CSR) - Pure Python implementation
  - SciPy CSR - Library-optimized implementation

- **Comprehensive Benchmarking:**
  - Matrix sizes: 64×64 to 2048×2048
  - Sparsity levels: 50%, 70%, 90%, 95%, 99%
  - Metrics: Execution time, peak memory usage, speedup analysis

- **Real-World Validation:**
  - mc2depi matrix (525,825×525,825, 99.9992% sparsity)

- **Visualization:**
  - 7 publication-ready plots (dense/sparse performance, memory comparison)
  - Automated CSV result generation


## 📁 Repository Structure

```
.
├── python/
│   └── src/
│       └── matrix/
│           ├── benchmark/                     # Benchmarks
│           │   ├── benchmark_dense_vs_sparse.py
│           │   ├── benchmark_dense.py
│           │   └── benchmark_sparse.py
│           ├── dense/                         # Dense implementations
│           │   ├── matrix_numpy.py
│           │   ├── matrix.py
│           │   └── utils.py
│           ├── plots/                         # Plot scripts
│           │   ├── plot_dense.py
│           │   └── plot_sparse.py
│           ├── sparse/                        # Sparse implementations
│           │   ├── matrix_csr.py
│           │   └── matrix_scipy.py
│           └── test/                          # Unit tests
│               └── matrix/
│                   ├── dense/
│                   │   ├── test_matrix.py
│                   │   └── test_matrix_numpy.py
│                   └── sparse/
│                       ├── test_matrix_csr.py
│                       └── test_mc2depi_matrix.py
├── results/                                   # CSV benchmark results
│   ├── dense_algorithms.csv
│   ├── sparse_algorithms.csv
│   └── dense_vs_sparse.csv
├── plots/                                     # Generated plots
│   ├── python_pure_time_dense.png
│   ├── numpy_time_dense.png
│   ├── python_vs_numpy_dense.png
│   ├── memory_comparison_dense.png
│   ├── sparse_pure_analysis.png
│   ├── sparse_scipy_analysis.png
│   └── sparse_comparison.png
├── mc2depi/
│   └── mc2depi.mtx
├── LICENSE
├── README.md
└── .gitignore
```
### 📂 Folder and File Descriptions

#### `python/src/matrix/`
- **benchmark/**: Benchmarking scripts for dense, sparse, and crossover analysis
- **dense/**: Pure Python and NumPy implementations of dense algorithms
- **sparse/**: CSR format implementations (Pure Python + SciPy)
- **plots/**: Visualization scripts for generating performance plots
- **test/matrix/**: Unit tests for all implementations

#### `results/`
- CSV files with detailed benchmark results (time, memory, speedup)

#### `plots/`
- Publication-ready PNG plots for paper inclusion

#### `mc2depi/`
- Real-world sparse matrix from SuiteSparse Matrix Collection


## 🛠️ Installation

### Requirements

- **Python 3.9+**
- **Libraries:** NumPy, SciPy, Pandas, Matplotlib, psutil
- **Terminal:** Linux, MacOS, or Windows
- **IDE (recommended):** VS Code

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pabcablan/Performance_Benchmark_of_matrix_multiplication.git
   cd Performance_Benchmark_of_matrix_multiplication
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv performance_bd_venv

   # Activate (Linux/Mac)
   source performance_bd_venv/bin/activate

   # Activate (Windows)
   performance_bd_venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy scipy pandas matplotlib psutil
   ```


## 🚀 Execution

### Dense Matrix Benchmarks

```bash
cd python
python src/matrix/benchmark/benchmark_dense.py <output_directory>
```

**Output:**
- `<output_directory>/dense_algorithms.csv` - Detailed results for all dense algorithms
- Console summary with key findings

### Sparse Matrix Benchmarks

```bash
cd python
python src/matrix/benchmark/benchmark_sparse.py <output_directory>
```

**Output:**
- `<output_directory>/sparse_algorithms.csv` - Results across sparsity levels
- Console summary with speedup analysis

### Dense vs Sparse Comparison

```bash
cd python
python src/matrix/benchmark/benchmark_dense_vs_sparse.py <output_directory>
```

**Output:**
- `<output_directory>/dense_vs_sparse.csv` - Crossover point analysis
- Console summary with threshold recommendations

### Real-World Validation (mc2depi)

```bash
# In source folder
python python/test/matrix/sparse/test_mc2depy_matrix.py
```

**Output:**
- Performance metrics for 525,825×525,825 sparse matrix
- Memory usage comparison vs dense representation


## 📊 Results and Visualization

### Generate All Plots

```bash
cd python

# Dense plots (4 figures)
python src/matrix/plots/plot_dense.py <dense_csv_path> <plot_directory>

# Sparse plots (3 figures)
python src/matrix/plots/plot_sparse.py <sparse_csv_path> <plot_directory>
```

**Generated Plots:**

1. **python_pure_time_dense.png** - Pure Python algorithm comparison
2. **numpy_time_dense.png** - NumPy algorithm comparison
3. **python_vs_numpy_dense.png** - Library advantage visualization
4. **memory_comparison_dense.png** - Memory usage (Pure Python vs NumPy)
5. **sparse_pure_analysis.png** - CSR-Pure: Time vs Sparsity + Size
6. **sparse_scipy_analysis.png** - SciPy: Time vs Sparsity + Size
7. **sparse_comparison.png** - CSR-Pure vs SciPy at 90% sparsity

**All plots are saved to `<plot_directory>`**


## 🧪 Testing

### Run All Unit Tests

```bash
cd python/test/matrix

# Dense tests
python dense/test_matrix.py
python dense/test_matrix_numpy.py

# Sparse tests
python sparse/test_matrix_csr.py
python sparse/test_mc2depi_matrix.py
```

**Tests verify:**
- ✅ Correctness of all algorithms
- ✅ Numerical accuracy (identity matrix, known results)
- ✅ Edge cases (small matrices, high sparsity)


## 🎓 Key Results Summary

| **Category** | **Best Implementation** | **Speedup** | **When to Use** |
|--------------|-------------------------|-------------|-----------------|
| Dense (any size) | NumPy-matmul | 8,550× | Always for dense matrices |
| Sparse (50% sparsity) | Dense-NumPy | 25× faster than SciPy | Low sparsity workloads |
| Sparse (70% sparsity) | CSR-Pure | 1.08× vs Dense-Python | Pure Python + moderate sparsity |
| Sparse (95% sparsity) | SciPy CSR | 7,826× vs CSR-Pure | Library-optimized + high sparsity |
| Sparse (99% sparsity) | SciPy CSR | 13,633× vs CSR-Pure | Extreme sparsity applications |

**Decision Guide:**
- **Pure Python:** Use sparse above **70% sparsity**
- **Libraries (NumPy/SciPy):** Use sparse above **95% sparsity**
- **Below thresholds:** Dense representations outperform sparse


## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
