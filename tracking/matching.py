from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the greedy matching algorithm.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids = [], col_ids = []
    return row_ids, col_ids


def hungarian_matching(cost_matrix: np.ndarray) -> Tuple[List, List]:
    """Perform matching based on the Hungarian matching algorithm.
    For simplicity, we just call the scipy `linear_sum_assignment` function. Please refer to
    https://en.wikipedia.org/wiki/Hungarian_algorithm and
    https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
    for more details of the hungarian matching implementation.

    Args:
        cost matrix of shape [M, N], where cost[i, j] is the cost of matching i to j
    Returns:
        (row_ids, col_ids), where row_ids and col_ids are lists of the same length,
        and each (row_ids[k], col_ids[k]) is a match.

        Example: if M = 3, N = 4, then the return values of ([0, 1, 2], [3, 1, 0]) means the final
        assignment corresponds to costs[0, 3], costs[1, 1] and costs[2, 0].
    """
    # TODO: Replace this stub code.
    row_ids = [], col_ids = []
    return row_ids, col_ids
