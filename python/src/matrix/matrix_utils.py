import random
import numpy as np
from scipy.sparse import csr_matrix
from python.src.matrix.sparsematrix.sparse_matrix_csr import SparseMatrixCSR
from python.src.matrix.sparsematrix.sparse_matrix_scipy import SparseMatrixSciPy



def add_matrices(A, B):
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def subtract_matrices(A, B):
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]


def generate_matrices(n):
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    return A, B

def generate_sparse_matrix_csr(n, sparsity=0.9):
    values = []
    col_index = []
    row_ptr = [0]
    
    for i in range(n):
        for j in range(n):
            if random.random() > sparsity:
                values.append(random.random())
                col_index.append(j)
        row_ptr.append(len(values))
    
    return SparseMatrixCSR(values, col_index, row_ptr, (n, n))

def csr_from_dense(dense_matrix):
    if not dense_matrix or not dense_matrix[0]:
        return SparseMatrixCSR([], [], [0], (0, 0))
    
    values = []
    col_index = []
    row_ptr = [0]
    
    n_rows = len(dense_matrix)
    n_cols = len(dense_matrix[0])
    
    for i in range(n_rows):
        for j in range(n_cols):
            if dense_matrix[i][j] != 0:
                values.append(dense_matrix[i][j])
                col_index.append(j)
        row_ptr.append(len(values))
    
    return SparseMatrixCSR(values, col_index, row_ptr, (n_rows, n_cols))


def csr_to_dense(sparse_matrix):
    n_rows, n_cols = sparse_matrix.shape
    result = [[0] * n_cols for _ in range(n_rows)]
    
    for i in range(n_rows):
        for k in range(sparse_matrix.row_ptr[i], sparse_matrix.row_ptr[i + 1]):
            result[i][sparse_matrix.col_index[k]] = sparse_matrix.values[k]
    
    return result


def scipy_from_dense(dense_matrix):
    return SparseMatrixSciPy(csr_matrix(dense_matrix))


def scipy_to_dense(sparse_matrix):
    return sparse_matrix.matrix.toarray()


def generate_sparse_matrix_scipy(n, sparsity=0.9):
    density = 1 - sparsity
    random_matrix = np.random.rand(n, n)
    mask = np.random.rand(n, n) < density
    sparse_data = random_matrix * mask
    return SparseMatrixSciPy(csr_matrix(sparse_data))