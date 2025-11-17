import random


class DenseMatrix:
    
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)
    
    @classmethod
    def random(cls, n):
        data = [[random.random() for _ in range(n)] for _ in range(n)]
        return cls(data)
    
    def multiply_standard(self, other):
        n = self.shape[0]
        C = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += self.data[i][k] * other.data[k][j]
        
        return DenseMatrix(C)
    
    def multiply_row_oriented(self, other):
        n = self.shape[0]
        C = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for k in range(n):
                aik = self.data[i][k]
                for j in range(n):
                    C[i][j] += aik * other.data[k][j]
        
        return DenseMatrix(C)
    
    def multiply_tiled(self, other, block_size=32):
        n = self.shape[0]
        C = [[0] * n for _ in range(n)]
        
        for i_block in range(0, n, block_size):
            for j_block in range(0, n, block_size):
                for k_block in range(0, n, block_size):
                    
                    i_limit = min(i_block + block_size, n)
                    j_limit = min(j_block + block_size, n)
                    k_limit = min(k_block + block_size, n)
                    
                    for i in range(i_block, i_limit):
                        for k in range(k_block, k_limit):
                            aik = self.data[i][k]
                            for j in range(j_block, j_limit):
                                C[i][j] += aik * other.data[k][j]
        
        return DenseMatrix(C)
    
    def multiply_strassen(self, other):
        def strassen_recursive(A, B):
            n = len(A)
            
            if n <= 64:
                C = [[0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            C[i][j] += A[i][k] * B[k][j]
                return C
            
            next_pow2 = 1
            while next_pow2 < n:
                next_pow2 *= 2
            
            if next_pow2 != n:
                A_padded = [[0] * next_pow2 for _ in range(next_pow2)]
                B_padded = [[0] * next_pow2 for _ in range(next_pow2)]
                
                for i in range(n):
                    for j in range(n):
                        A_padded[i][j] = A[i][j]
                        B_padded[i][j] = B[i][j]
                
                C_padded = strassen_recursive(A_padded, B_padded)
                return [[C_padded[i][j] for j in range(n)] for i in range(n)]
            
            mid = n // 2
            
            A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
            A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
            A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
            A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
            
            B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
            B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
            B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
            B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
            
            def add_matrices(X, Y):
                n = len(X)
                return [[X[i][j] + Y[i][j] for j in range(n)] for i in range(n)]
            
            def sub_matrices(X, Y):
                n = len(X)
                return [[X[i][j] - Y[i][j] for j in range(n)] for i in range(n)]
            
            M1 = strassen_recursive(add_matrices(A11, A22), add_matrices(B11, B22))
            M2 = strassen_recursive(add_matrices(A21, A22), B11)
            M3 = strassen_recursive(A11, sub_matrices(B12, B22))
            M4 = strassen_recursive(A22, sub_matrices(B21, B11))
            M5 = strassen_recursive(add_matrices(A11, A12), B22)
            M6 = strassen_recursive(sub_matrices(A21, A11), add_matrices(B11, B12))
            M7 = strassen_recursive(sub_matrices(A12, A22), add_matrices(B21, B22))
            
            C11 = add_matrices(sub_matrices(add_matrices(M1, M4), M5), M7)
            C12 = add_matrices(M3, M5)
            C21 = add_matrices(M2, M4)
            C22 = add_matrices(sub_matrices(add_matrices(M1, M3), M2), M6)
            
            C = [[0] * n for _ in range(n)]
            for i in range(mid):
                for j in range(mid):
                    C[i][j] = C11[i][j]
                    C[i][j + mid] = C12[i][j]
                    C[i + mid][j] = C21[i][j]
                    C[i + mid][j + mid] = C22[i][j]
            
            return C
        
        return DenseMatrix(strassen_recursive(self.data, other.data))