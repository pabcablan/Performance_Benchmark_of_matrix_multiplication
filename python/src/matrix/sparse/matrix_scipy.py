import numpy as np
from scipy.sparse import csr_matrix


class SparseMatrixSciPy:
    
    def __init__(self, scipy_matrix):
        self.matrix = scipy_matrix
        self.shape = scipy_matrix.shape
    
    @classmethod
    def from_dense(cls, dense_matrix):
        return cls(csr_matrix(dense_matrix))
    
    @classmethod
    def random(cls, n, sparsity=0.9):
        density = 1 - sparsity
        random_matrix = np.random.rand(n, n)
        mask = np.random.rand(n, n) < density
        sparse_data = random_matrix * mask
        return cls(csr_matrix(sparse_data))
    
    def multiply(self, other):
        result_matrix = self.matrix @ other.matrix
        return SparseMatrixSciPy(result_matrix)
    
    def to_dense(self):
        return self.matrix.toarray()
    
    def numbers_non_zero(self):
        return self.matrix.nnz
    
    def get_sparsity(self):
        total = self.matrix.shape[0] * self.matrix.shape[1]
        return (total - self.numbers_non_zero()) / total if total > 0 else 0