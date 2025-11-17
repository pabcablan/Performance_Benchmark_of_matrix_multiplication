import unittest
import numpy as np
from python.src.matrix.dense.matrix_numpy import DenseMatrixNumPy
from python.src.matrix.dense.utils import generate_matrices_numpy


class TestDenseMatrixNumPy(unittest.TestCase):

    def setUp(self):
        self.A = DenseMatrixNumPy([[1, 2], [3, 4]])
        self.B = DenseMatrixNumPy([[5, 6], [7, 8]])
        self.expected = np.array([[19, 22], [43, 50]])

    def test_multiply_builtin(self):
        result = self.A.multiply_builtin(self.B)
        np.testing.assert_array_almost_equal(result.data, self.expected)

    def test_multiply_matmul(self):
        result = self.A.multiply_matmul(self.B)
        np.testing.assert_array_almost_equal(result.data, self.expected)

    def test_multiply_tiled(self):
        for block_size in [1, 2, 4]:
            with self.subTest(block_size=block_size):
                result = self.A.multiply_tiled(self.B, block_size)
                np.testing.assert_array_almost_equal(result.data, self.expected)

    def test_multiply_strassen(self):
        result = self.A.multiply_strassen(self.B)
        np.testing.assert_array_almost_equal(result.data, self.expected)

    def test_strassen_any_size(self):
        A = DenseMatrixNumPy([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = DenseMatrixNumPy([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        
        result_strassen = A.multiply_strassen(B)
        result_builtin = A.multiply_builtin(B)
        
        np.testing.assert_array_almost_equal(result_strassen.data, result_builtin.data, decimal=10)

    def test_algorithms_equivalence(self):
        n = 8
        A_np, B_np = generate_matrices_numpy(n)
        A = DenseMatrixNumPy(A_np)
        B = DenseMatrixNumPy(B_np)

        result_builtin = A.multiply_builtin(B)
        result_matmul = A.multiply_matmul(B)
        result_tiled = A.multiply_tiled(B)
        result_strassen = A.multiply_strassen(B)

        np.testing.assert_array_almost_equal(result_builtin.data, result_matmul.data, decimal=10)
        np.testing.assert_array_almost_equal(result_builtin.data, result_tiled.data, decimal=10)
        np.testing.assert_array_almost_equal(result_builtin.data, result_strassen.data, decimal=10)

    def test_random(self):
        A = DenseMatrixNumPy.random(10)
        B = DenseMatrixNumPy.random(10)
        C = A.multiply_builtin(B)
        self.assertEqual(C.shape, (10, 10))


if __name__ == '__main__':
    unittest.main(verbosity=2)