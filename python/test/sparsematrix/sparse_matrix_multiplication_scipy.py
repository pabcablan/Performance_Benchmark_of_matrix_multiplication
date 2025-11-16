import unittest
import numpy as np
from python.src.matrix.matrix_utils import scipy_from_dense, scipy_to_dense, generate_sparse_matrix_scipy

class TestSparseMatrixSciPy(unittest.TestCase):
    
    def test_from_dense(self):
        dense = [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        sparse = scipy_from_dense(dense)
        
        self.assertEqual(sparse.shape, (3, 3))
        self.assertEqual(sparse.nnz(), 5)
    
    def test_multiply_simple(self):
        A = scipy_from_dense([[1, 2], [3, 4]])
        B = scipy_from_dense([[5, 6], [7, 8]])
        
        C = A.multiply(B)
        
        expected = np.array([[19, 22], [43, 50]])
        
        np.testing.assert_array_equal(scipy_to_dense(C), expected)
    
    def test_multiply_sparse(self):
        A = scipy_from_dense([[0, 5, 0], [0, 0, 8], [1, 0, 0]])
        B = scipy_from_dense([[0, 0, 2], [3, 0, 0], [0, 4, 0]])
        
        C = A.multiply(B)
        
        expected = np.array([[15, 0, 0], [0, 32, 0], [0, 0, 2]])
        
        np.testing.assert_array_equal(scipy_to_dense(C), expected)
    
    def test_generate_random(self):
        for sparsity in [0.5, 0.9, 0.95]:
            with self.subTest(sparsity=sparsity):
                matrix = generate_sparse_matrix_scipy(100, sparsity)
                
                actual_sparsity = matrix.get_sparsity()
                self.assertGreater(actual_sparsity, sparsity - 0.1)
                self.assertLess(actual_sparsity, sparsity + 0.1)
    
    def test_nnz_and_sparsity(self):
        dense = [[0, 5, 0, 0], [0, 0, 8, 0], [0, 0, 0, 3], [1, 0, 0, 0]]
        sparse = scipy_from_dense(dense)
        
        self.assertEqual(sparse.nnz(), 4)
        self.assertAlmostEqual(sparse.get_sparsity(), 0.75)


if __name__ == '__main__':
    unittest.main(verbosity=2)