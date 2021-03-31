import torch.nn as nn
import torch
from tag_op.tagop.tools.allennlp import replace_masked_values, masked_log_softmax


class SingleSpanHead(nn.Module):

    def __init__(self, input_size):
        super(SingleSpanHead, self).__init__()
        self.start_pos_predict = nn.Linear(input_size, 1)
        self.end_pos_predict = nn.Linear(input_size, 1)
        self.NLL = nn.NLLLoss(reduction="sum")

    def forward(self, input_vec: torch.LongTensor, mask: torch.LongTensor, label: torch.LongTensor):
        start_logits = self.start_pos_predict(input_vec).squeeze(-1)
        end_logits = self.end_pos_predict(input_vec).squeeze(-1)

        start_log_probs = masked_log_softmax(start_logits, mask)
        end_log_probs = masked_log_softmax(end_logits, mask)

        # Info about the best span prediction
        start_logits = replace_masked_values(start_logits, mask, -1e7)
        end_logits = replace_masked_values(end_logits, mask, -1e7)

        # Shape: (batch_size, 2)
        best_span = get_best_span(start_logits, end_logits)
        if torch.LongTensor([-1, -1]) in label:
            label_fit_index = label[:, 0] != - 1
            label = label[label_fit_index]
            start_log_probs = start_log_probs[label_fit_index]
            end_log_probs = end_log_probs[label_fit_index]
        loss = self.NLL(start_log_probs, label[:, 0]) + self.NLL(end_log_probs, label[:, 1])

        return loss, best_span


def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.

    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                          device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)
