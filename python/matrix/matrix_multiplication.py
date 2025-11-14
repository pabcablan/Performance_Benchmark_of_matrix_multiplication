from python.matrix.matrix_utils import subtract_matrices, add_matrices


def matrix_multiply_standard(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_multiply_row_oriented(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C


def matrix_multiply_tiled(A, B, block_size=32):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]

    for i_block in range(0, n, block_size):
        for j_block in range(0, n, block_size):
            for k_block in range(0, n, block_size):

                i_limit = min(i_block + block_size, n)
                j_limit = min(j_block + block_size, n)
                k_limit = min(k_block + block_size, n)

                for i in range(i_block, i_limit):
                    for k in range(k_block, k_limit):
                        aik = A[i][k]
                        for j in range(j_block, j_limit):
                            C[i][j] += aik * B[k][j]
    return C

def strassen(A, B):
    n = len(A)
    
    if n <= 64:
        return matrix_multiply_standard(A, B)
    

    next_pow2 = 1
    while next_pow2 < n:
        next_pow2 *= 2
    
    if next_pow2 != n:
        A_padded = [[0 for _ in range(next_pow2)] for _ in range(next_pow2)]
        B_padded = [[0 for _ in range(next_pow2)] for _ in range(next_pow2)]
        
        for i in range(n):
            for j in range(n):
                A_padded[i][j] = A[i][j]
                B_padded[i][j] = B[i][j]
        
        C_padded = strassen(A_padded, B_padded)
        
        C = [[C_padded[i][j] for j in range(n)] for i in range(n)]
        return C
    
    mid = n // 2
    
    A11 = [[A[i][j] for j in range(mid)] for i in range(mid)]
    A12 = [[A[i][j] for j in range(mid, n)] for i in range(mid)]
    A21 = [[A[i][j] for j in range(mid)] for i in range(mid, n)]
    A22 = [[A[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    B11 = [[B[i][j] for j in range(mid)] for i in range(mid)]
    B12 = [[B[i][j] for j in range(mid, n)] for i in range(mid)]
    B21 = [[B[i][j] for j in range(mid)] for i in range(mid, n)]
    B22 = [[B[i][j] for j in range(mid, n)] for i in range(mid, n)]
    
    M1 = strassen(add_matrices(A11, A22), add_matrices(B11, B22))
    M2 = strassen(add_matrices(A21, A22), B11)
    M3 = strassen(A11, subtract_matrices(B12, B22))
    M4 = strassen(A22, subtract_matrices(B21, B11))
    M5 = strassen(add_matrices(A11, A12), B22)
    M6 = strassen(subtract_matrices(A21, A11), add_matrices(B11, B12))
    M7 = strassen(subtract_matrices(A12, A22), add_matrices(B21, B22))
    
    C11 = add_matrices(subtract_matrices(add_matrices(M1, M4), M5), M7)
    C12 = add_matrices(M3, M5)
    C21 = add_matrices(M2, M4)
    C22 = add_matrices(subtract_matrices(add_matrices(M1, M3), M2), M6)
    
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(mid):
        for j in range(mid):
            C[i][j] = C11[i][j]
            C[i][j + mid] = C12[i][j]
            C[i + mid][j] = C21[i][j]
            C[i + mid][j + mid] = C22[i][j]
    
    return C