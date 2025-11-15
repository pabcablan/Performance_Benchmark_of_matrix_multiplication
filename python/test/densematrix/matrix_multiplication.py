import unittest
import python.src.matrix.densematrix.matrix_multiplication as mm
import python.src.matrix.matrix_utils as mu

class TestMatrixMultiplication(unittest.TestCase):

    def setUp(self):
        self.A = [[1, 2], [3, 4]]
        self.B = [[5, 6], [7, 8]]
        self.expected = [[19, 22], [43, 50]]

    def test_standard_multiplication(self):
        result = mm.matrix_multiply_standard(self.A, self.B)
        self.assertEqual(result, self.expected)

    def test_row_oriented_multiplication(self):
        result = mm.matrix_multiply_row_oriented(self.A, self.B)
        self.assertEqual(result, self.expected)

    def test_tiled_multiplication(self):
        for block_size in [1, 2, 4]:
            with self.subTest(block_size=block_size):
                result = mm.matrix_multiply_tiled(self.A, self.B, block_size)
                self.assertEqual(result, self.expected)

    def test_strassen_multiplication(self):
        result = mm.strassen(self.A, self.B)
        self.assertEqual(result, self.expected)

    def test_strassen_any_size(self):
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        
        result_strassen = mm.strassen(A, B)
        result_standard = mm.matrix_multiply_standard(A, B)
        
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(result_strassen[i][j], result_standard[i][j], places=10)

    def test_algorithms_equivalence(self):
        n = 8
        A, B = mu.generate_matrices(n)

        result_standard = mm.matrix_multiply_standard(A, B)
        result_row = mm.matrix_multiply_row_oriented(A, B)
        result_tiled = mm.matrix_multiply_tiled(A, B)
        result_strassen = mm.strassen(A, B)

        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(result_standard[i][j], result_row[i][j], places=10)
                self.assertAlmostEqual(result_standard[i][j], result_tiled[i][j], places=10)
                self.assertAlmostEqual(result_standard[i][j], result_strassen[i][j], places=10)

if __name__ == '__main__':
    unittest.main(verbosity=2)