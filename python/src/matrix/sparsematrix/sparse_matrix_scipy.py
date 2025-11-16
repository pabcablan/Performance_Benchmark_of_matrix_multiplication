class SparseMatrixSciPy:
    
    def __init__(self, scipy_matrix):
        self.matrix = scipy_matrix
    
    def multiply(self, other):
        result_matrix = self.matrix @ other.matrix
        return SparseMatrixSciPy(result_matrix)
    
    def nnz(self):
        return self.matrix.nnz
    
    def get_sparsity(self):
        total = self.matrix.shape[0] * self.matrix.shape[1]
        return (total - self.nnz()) / total if total > 0 else 0
    
    @property
    def shape(self):
        return self.matrix.shape