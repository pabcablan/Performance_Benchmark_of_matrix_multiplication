import unittest
from python.src.matrix.sparse.matrix_csr import SparseMatrixCSR


class TestSparseMatrixCSR(unittest.TestCase):
    
    def test_multiply(self):
        A = SparseMatrixCSR.from_dense([[0, 5, 0], [0, 0, 8], [1, 0, 0]])
        B = SparseMatrixCSR.from_dense([[0, 0, 2], [3, 0, 0], [0, 4, 0]])
        
        C = A.multiply(B)
        
        self.assertEqual(C.shape, (3, 3))
        
        self.assertEqual(C.numbers_non_zero(), 3)

        self.assertEqual(C.values, [15, 32, 2])
        self.assertEqual(C.col_index, [0, 1, 2])
        self.assertEqual(C.row_ptr, [0, 1, 2, 3])
    
    def test_numbers_non_zero(self):
        A = SparseMatrixCSR.from_dense([[0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3], [1, 0, 0, 0]])
        self.assertEqual(A.numbers_non_zero(), 4)
    
    def test_sparsity(self):
        A = SparseMatrixCSR.from_dense([[0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3], [1, 0, 0, 0]])
        self.assertAlmostEqual(A.get_sparsity(), 0.75)
    
    def test_random(self):
        A = SparseMatrixCSR.random(100, sparsity=0.9)
        
        self.assertEqual(A.shape, (100, 100))
        self.assertGreater(A.get_sparsity(), 0.8)
        self.assertLess(A.get_sparsity(), 1.0)
    
    def test_multiply_random(self):
        A = SparseMatrixCSR.random(50, sparsity=0.9)
        B = SparseMatrixCSR.random(50, sparsity=0.9)
        
        C = A.multiply(B)
        
        self.assertEqual(C.shape, (50, 50))
        self.assertGreater(C.numbers_non_zero(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)