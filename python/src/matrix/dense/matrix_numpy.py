import numpy as np


class DenseMatrixNumPy:
    
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
    
    @classmethod
    def random(cls, n):
        return cls(np.random.rand(n, n))
    
    def multiply_builtin(self, other):
        result = np.dot(self.data, other.data)
        return DenseMatrixNumPy(result)
    
    def multiply_matmul(self, other):
        result = self.data @ other.data
        return DenseMatrixNumPy(result)
    
    def multiply_tiled(self, other, block_size=32):
        n = self.data.shape[0]
        C = np.zeros((n, n))
        
        for i_block in range(0, n, block_size):
            for j_block in range(0, n, block_size):
                for k_block in range(0, n, block_size):
                    
                    i_end = min(i_block + block_size, n)
                    j_end = min(j_block + block_size, n)
                    k_end = min(k_block + block_size, n)
                    
                    C[i_block:i_end, j_block:j_end] += np.dot(
                        self.data[i_block:i_end, k_block:k_end],
                        other.data[k_block:k_end, j_block:j_end]
                    )
        
        return DenseMatrixNumPy(C)
    
    def multiply_strassen(self, other):
        def strassen_recursive(A, B):
            n = A.shape[0]
            
            if n <= 64:
                return np.dot(A, B)
            
            next_pow2 = 1
            while next_pow2 < n:
                next_pow2 *= 2
            
            if next_pow2 != n:
                A_padded = np.zeros((next_pow2, next_pow2))
                B_padded = np.zeros((next_pow2, next_pow2))
                
                A_padded[:n, :n] = A
                B_padded[:n, :n] = B
                
                C_padded = strassen_recursive(A_padded, B_padded)
                return C_padded[:n, :n]
            
            mid = n // 2
            
            A11 = A[:mid, :mid]
            A12 = A[:mid, mid:]
            A21 = A[mid:, :mid]
            A22 = A[mid:, mid:]
            
            B11 = B[:mid, :mid]
            B12 = B[:mid, mid:]
            B21 = B[mid:, :mid]
            B22 = B[mid:, mid:]
            
            M1 = strassen_recursive(A11 + A22, B11 + B22)
            M2 = strassen_recursive(A21 + A22, B11)
            M3 = strassen_recursive(A11, B12 - B22)
            M4 = strassen_recursive(A22, B21 - B11)
            M5 = strassen_recursive(A11 + A12, B22)
            M6 = strassen_recursive(A21 - A11, B11 + B12)
            M7 = strassen_recursive(A12 - A22, B21 + B22)
            
            C11 = M1 + M4 - M5 + M7
            C12 = M3 + M5
            C21 = M2 + M4
            C22 = M1 + M3 - M2 + M6
            
            C = np.vstack([np.hstack([C11, C12]), np.hstack([C21, C22])])
            return C
        
        result = strassen_recursive(self.data, other.data)
        return DenseMatrixNumPy(result)