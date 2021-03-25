from typing import Dict, Any

import torch

class Head(torch.nn.Module):
    def gold_log_marginal_likelihood(self, *args: torch.LongTensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def decode_answer(self, *args: torch.LongTensor, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError