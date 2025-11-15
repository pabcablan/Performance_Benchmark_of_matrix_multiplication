import unittest
from python.src.matrix.sparsematrix.sparse_matrix_csr import SparseMatrixCSR
from python.src.matrix.matrix_utils import csr_from_dense, csr_to_dense, generate_sparse_matrix_csr


class TestSparseMatrixCSR(unittest.TestCase):
    
    def test_multiply(self):
        A = csr_from_dense([[0, 5, 0], [0, 0, 8], [1, 0, 0]])
        B = csr_from_dense([[0, 0, 2], [3, 0, 0], [0, 4, 0]])
        
        C = A.multiply(B)
        
        expected = [[15, 0, 0], [0, 32, 0], [0, 0, 2]]
        
        self.assertEqual(csr_to_dense(C), expected)
    
    def test_non_equal_zero(self):
        A = csr_from_dense([[0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3], [1, 0, 0, 0]])
        self.assertEqual(A.nnz(), 4)
    
    def test_sparsity(self):
        A = csr_from_dense([[0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3], [1, 0, 0, 0]])
        self.assertAlmostEqual(A.get_sparsity(), 0.75)
    
    def test_generate(self):
        A = generate_sparse_matrix_csr(100, sparsity=0.9)
        
        self.assertEqual(A.shape, (100, 100))
        self.assertAlmostEqual(A.get_sparsity(), 0.9, delta=1E-3)


if __name__ == '__main__':
    unittest.main(verbosity=2)