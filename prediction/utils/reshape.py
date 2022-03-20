from typing import List, Tuple

import torch
from torch import Tensor


def flatten(values_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
    batch_ids = []
    for batch_id, values in enumerate(values_list):
        batch_ids.extend([batch_id] * len(values))
    return torch.cat(values_list), torch.tensor(batch_ids, dtype=int)


def unflatten_batch(values: Tensor, batch_ids: Tensor) -> List[Tensor]:
    values_list = []
    num_batches = max(batch_ids)
    for batch_id in range(num_batches + 1):
        values_list.append(values[batch_ids == batch_id])
    return values_list
