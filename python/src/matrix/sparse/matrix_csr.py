import random


class SparseMatrixCSR:
    
    def __init__(self, values, col_index, row_ptr, shape):
        self.values = values
        self.col_index = col_index
        self.row_ptr = row_ptr
        self.shape = shape
    
    @classmethod
    def from_dense(cls, dense_matrix):
        if not dense_matrix or not dense_matrix[0]:
            return cls([], [], [0], (0, 0))
        
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
        
        return cls(values, col_index, row_ptr, (n_rows, n_cols))
    
    @classmethod
    def random(cls, n, sparsity=0.9):
        values = []
        col_index = []
        row_ptr = [0]
        
        for i in range(n):
            for j in range(n):
                if random.random() > sparsity:
                    values.append(random.random())
                    col_index.append(j)
            row_ptr.append(len(values))
        
        return cls(values, col_index, row_ptr, (n, n))
    
    def multiply(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Incompatible Dimensions: {self.shape} × {other.shape}")
        
        n_rows = self.shape[0]
        n_cols = other.shape[1]
        
        result_dict = {}
        
        for i in range(n_rows):
            for idx_a in range(self.row_ptr[i], self.row_ptr[i + 1]):
                k = self.col_index[idx_a]
                a_val = self.values[idx_a]
                
                for idx_b in range(other.row_ptr[k], other.row_ptr[k + 1]):
                    j = other.col_index[idx_b]
                    b_val = other.values[idx_b]
                    
                    if (i, j) not in result_dict:
                        result_dict[(i, j)] = 0
                    result_dict[(i, j)] += a_val * b_val
        
        values = []
        col_index = []
        row_ptr = [0]
        
        for i in range(n_rows):
            row_elements = sorted([(j, val) for (row, j), val in result_dict.items() if row == i])
            for j, val in row_elements:
                values.append(val)
                col_index.append(j)
            row_ptr.append(len(values))
        
        return SparseMatrixCSR(values, col_index, row_ptr, (n_rows, n_cols))
    
    def to_dense(self):
        n_rows, n_cols = self.shape
        dense = [[0] * n_cols for _ in range(n_rows)]
        
        for i in range(n_rows):
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_index[idx]
                dense[i][j] = self.values[idx]
        
        return dense

    def numbers_non_zero(self):
        return len(self.values)
    
    def get_sparsity(self):
        total = self.shape[0] * self.shape[1]
        return (total - self.numbers_non_zero()) / total if total > 0 else 0