import numpy as np
import pytest

from tracking.matching import greedy_matching, hungarian_matching


class Helpers:
    @staticmethod
    def match(cost_matrix: np.ndarray):
        M, N = cost_matrix.shape[0], cost_matrix.shape[1]
        greedy_matrix, hungarian_matrix = np.zeros((M, N)), np.zeros((M, N))

        row_ids, col_ids = greedy_matching(cost_matrix)
        greedy_matrix[row_ids, col_ids] = 1
        greedy_cost = np.sum(cost_matrix * greedy_matrix)
        row_ids, col_ids = hungarian_matching(cost_matrix)
        hungarian_matrix[row_ids, col_ids] = 1
        hungarian_cost = np.sum(cost_matrix * hungarian_matrix)

        return greedy_matrix, greedy_cost, hungarian_matrix, hungarian_cost


@pytest.fixture
def helpers():
    return Helpers


def test_same_solution(helpers):
    """Test example (greedy and hungarian have same solutions)"""
    cost_matrix = np.array([[15, 2, 3], [2, 7, 5], [5, 8, 3]])
    greedy_matrix, greedy_cost, hungarian_matrix, hungarian_cost = helpers.match(
        cost_matrix
    )
    assert np.linalg.matrix_rank(greedy_matrix) == 3
    assert np.linalg.matrix_rank(hungarian_matrix) == 3
    assert np.sum(greedy_matrix) == 3
    assert np.sum(hungarian_matrix) == 3
    np.testing.assert_allclose(greedy_matrix, hungarian_matrix)
    np.testing.assert_allclose(greedy_cost, 7.0)


def test_diff_solution(helpers):
    """Test example (greedy and hungarian have different solutions)"""
    cost_matrix = np.array(
        [
            [0.43, 0.33, 0.15, 0.16],
            [0.75, 0.54, 0.62, 0.90],
            [0.65, 0.15, 0.45, 0.33],
        ]
    )
    greedy_matrix, greedy_cost, hungarian_matrix, hungarian_cost = helpers.match(
        cost_matrix
    )
    assert np.linalg.matrix_rank(greedy_matrix) == 3
    assert np.linalg.matrix_rank(hungarian_matrix) == 3
    assert np.sum(greedy_matrix) == 3
    assert np.sum(hungarian_matrix) == 3
    assert not np.isclose(greedy_cost, hungarian_cost)
