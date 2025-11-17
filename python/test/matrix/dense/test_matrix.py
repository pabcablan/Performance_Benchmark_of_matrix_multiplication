import unittest
from python.src.matrix.dense.matrix import DenseMatrix
from python.src.matrix.dense.utils import generate_matrices


class TestDenseMatrix(unittest.TestCase):

    def setUp(self):
        self.A = DenseMatrix([[1, 2], [3, 4]])
        self.B = DenseMatrix([[5, 6], [7, 8]])
        self.expected_data = [[19, 22], [43, 50]]

    def test_multiply_standard(self):
        result = self.A.multiply_standard(self.B)
        self.assertEqual(result.data, self.expected_data)

    def test_multiply_row_oriented(self):
        result = self.A.multiply_row_oriented(self.B)
        self.assertEqual(result.data, self.expected_data)

    def test_multiply_tiled(self):
        for block_size in [1, 2, 4]:
            with self.subTest(block_size=block_size):
                result = self.A.multiply_tiled(self.B, block_size)
                self.assertEqual(result.data, self.expected_data)

    def test_multiply_strassen(self):
        result = self.A.multiply_strassen(self.B)
        self.assertEqual(result.data, self.expected_data)

    def test_strassen_any_size(self):
        A = DenseMatrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        B = DenseMatrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        
        result_strassen = A.multiply_strassen(B)
        result_standard = A.multiply_standard(B)
        
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(result_strassen.data[i][j], result_standard.data[i][j], places=10)

    def test_algorithms_equivalence(self):
        n = 8
        A_list, B_list = generate_matrices(n)
        A = DenseMatrix(A_list)
        B = DenseMatrix(B_list)

        result_standard = A.multiply_standard(B)
        result_row = A.multiply_row_oriented(B)
        result_tiled = A.multiply_tiled(B)
        result_strassen = A.multiply_strassen(B)

        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(result_standard.data[i][j], result_row.data[i][j], places=10)
                self.assertAlmostEqual(result_standard.data[i][j], result_tiled.data[i][j], places=10)
                self.assertAlmostEqual(result_standard.data[i][j], result_strassen.data[i][j], places=10)

    def test_random(self):
        A = DenseMatrix.random(10)
        B = DenseMatrix.random(10)
        C = A.multiply_standard(B)
        self.assertEqual(C.shape, (10, 10))


if __name__ == '__main__':
    unittest.main(verbosity=2)