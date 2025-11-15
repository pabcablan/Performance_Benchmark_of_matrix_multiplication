import unittest
import numpy as np
import python.src.matrix.densematrix.matrix_multiplication_numpy as mm_np
import python.src.matrix.matrix_utils as mu


class TestMatrixMultiplicationNumPy(unittest.TestCase):

    def setUp(self):
        self.A = [[1, 2], [3, 4]]
        self.B = [[5, 6], [7, 8]]
        self.expected = np.array([[19, 22], [43, 50]])

    def test_numpy_builtin(self):
        result = mm_np.matrix_multiply_numpy_builtin(self.A, self.B)
        np.testing.assert_array_almost_equal(result, self.expected)

    def test_numpy_matmul(self):
        result = mm_np.matrix_multiply_numpy_matmul(self.A, self.B)
        np.testing.assert_array_almost_equal(result, self.expected)

    def test_numpy_vectorized(self):
        result = mm_np.matrix_multiply_numpy_vectorized(self.A, self.B)
        np.testing.assert_array_almost_equal(result, self.expected)

    def test_tiled_multiplication_numpy(self):
        for block_size in [1, 2, 4]:
            with self.subTest(block_size=block_size):
                result = mm_np.matrix_multiply_tiled_numpy(self.A, self.B, block_size)
                np.testing.assert_array_almost_equal(result, self.expected)

    def test_strassen_numpy(self):
        result = mm_np.strassen_numpy(self.A, self.B)
        np.testing.assert_array_almost_equal(result, self.expected)

    def test_strassen_numpy_any_size(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        
        result_strassen = mm_np.strassen_numpy(A, B)
        result_standard = mm_np.matrix_multiply_numpy_builtin(A, B)
        
        np.testing.assert_array_almost_equal(result_strassen, result_standard, decimal=10)

    def test_algorithms_equivalence_numpy(self):
        n = 8
        A, B = mu.generate_matrices_numpy(n)

        result_builtin = mm_np.matrix_multiply_numpy_builtin(A, B)
        result_matmul = mm_np.matrix_multiply_numpy_matmul(A, B)
        result_vectorized = mm_np.matrix_multiply_numpy_vectorized(A, B)
        result_tiled = mm_np.matrix_multiply_tiled_numpy(A, B)
        result_strassen = mm_np.strassen_numpy(A, B)

        np.testing.assert_array_almost_equal(result_builtin, result_matmul, decimal=10)
        np.testing.assert_array_almost_equal(result_builtin, result_vectorized, decimal=10)
        np.testing.assert_array_almost_equal(result_builtin, result_tiled, decimal=10)
        np.testing.assert_array_almost_equal(result_builtin, result_strassen, decimal=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)