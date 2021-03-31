import torch
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
import torch.nn.functional as F
from .heads.single_span_head import SingleSpanHead
from .heads.sequence_span_head import SequenceTagHead
from .newmo_util import Default_FNN
from .tools import allennlp as util
from typing import Dict, List, Tuple
from .file_utils import is_scatter_available
import numpy as np
from tag_op.data_builder.data_util import get_op_1, get_op_2, get_op_3, SCALE, OPERATOR_CLASSES_
import torch.nn.functional as F


class MutiHeadModel(nn.Module):
    def __init__(self,
                 bert,
                 config,
                 operator_classes: int,
                 scale_classes: int,
                 head_count: int = 5,
                 hidden_size: int = None,
                 dropout_prob: float = None,
                 arithmetic_op_index: List = None,
                 op_mode: int = None,
                 ablation_mode: int = None,
                 ):
        super(MutiHeadModel, self).__init__()
        self.tapas = bert
        self.config = config
        self.operator_classes = operator_classes
        self.scale_classes = scale_classes
        self._metrics = TaTQAEmAndF1(mode=2)
        if hidden_size is None:
            hidden_size = self.config.hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob
        self.head_count = head_count
        self.NLLLoss = nn.NLLLoss(reduction="sum")

        self.paragraph_summary_vector_module = nn.Linear(hidden_size, 1)
        self.table_summary_vector_module = nn.Linear(hidden_size, 1)
        self.question_summary_vector_module = nn.Linear(hidden_size, 1)

        self.head_predictor = Default_FNN(3 * hidden_size, hidden_size, head_count, dropout_prob)

        self.single_span_head = SingleSpanHead(hidden_size)
        self.sequence_tag_head = SequenceTagHead(hidden_size, dropout_prob)

    def heads_indices(self):
        return list(self._heads.keys())

    def summary_vector(self, encoding, mask, in_type='paragraph'):

        if in_type == 'paragraph':
            # Shape: (batch_size, seqlen)
            alpha = self.paragraph_summary_vector_module(encoding).squeeze()
        elif in_type == 'question':
            # Shape: (batch_size, seqlen)
            alpha = self.question_summary_vector_module(encoding).squeeze()
        elif in_type == 'table':
            alpha = self.table_summary_vector_module(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
        # Shape: (batch_size, seqlen)
        # (batch_size, #num of numbers, seqlen) for numbers
        alpha = util.masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        # (batch_size, #num of numbers, out) for numbers
        h = util.weighted_sum(encoding, alpha)
        return h

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                token_type_ids: torch.LongTensor,
                paragraph_mask: torch.LongTensor,
                paragraph_index: torch.LongTensor,
                tag_labels: torch.LongTensor,
                operator_labels: torch.LongTensor,
                scale_labels: torch.LongTensor,
                number_order_labels: torch.LongTensor,
                gold_answers: str,
                paragraph_tokens: List[List[str]],
                paragraph_numbers: List[np.ndarray],
                table_cell_numbers: List[np.ndarray],
                question_ids: List[str],
                span_pos_labels: torch.LongTensor,
                position_ids: torch.LongTensor = None,
                table_mask: torch.LongTensor = None,
                table_cell_index: torch.LongTensor = None,
                table_cell_tokens: List[List[str]] = None,
                mode=None,
                epoch=None, ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        token_representations = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)[0]
        batch_size = token_representations.shape[0]
        question_mask = attention_mask - paragraph_mask - table_mask
        question_mask[0] = 0

        paragraph_summary = self.summary_vector(token_representations, paragraph_mask, 'paragraph')
        table_summary = self.summary_vector(token_representations, table_mask, 'table')
        question_summary = self.summary_vector(token_representations, question_mask, 'question')

        operator_labels[operator_labels > 4] = 4
        paragraph_span_head_index = (operator_labels.squeeze() == 0).cpu().detach()
        table_span_head_index = (operator_labels.squeeze() == 1).cpu().detach()
        sequence_tag_head_index = (operator_labels.squeeze() > 1).cpu().detach()

        answer_head_logits = self.head_predictor(torch.cat([paragraph_summary,
                                                            table_summary, question_summary], dim=-1))
        answer_head_log_probs = F.log_softmax(answer_head_logits, -1)
        predict_head = torch.argmax(answer_head_log_probs, dim=-1).unsqueeze(-1)

        sequence_tag_head_loss = self.sequence_tag_head(token_representations[sequence_tag_head_index],
                                                        table_mask[sequence_tag_head_index],
                                                        paragraph_mask[sequence_tag_head_index],
                                                        tag_labels[sequence_tag_head_index])
        table_span_head_loss, table_span = self.single_span_head(token_representations[table_span_head_index],
                                                                 table_mask[table_span_head_index],
                                                                 span_pos_labels[table_span_head_index])
        paragraph_span_head_loss, paragraph_span = self.single_span_head(
            token_representations[paragraph_span_head_index],
            paragraph_mask[paragraph_span_head_index],
            span_pos_labels[paragraph_span_head_index])

        head_loss = self.NLLLoss(answer_head_log_probs, operator_labels)
        loss = head_loss + paragraph_span_head_loss + table_span_head_loss + sequence_tag_head_loss
        head_acc = (predict_head == operator_labels).float().mean()
        output_dict['loss'] = loss
        output_dict['head_acc'] = head_acc
        return output_dict

    def predict(self):
        pass

    def reset(self):
        self._metrics.reset()

    def set_metrics_mdoe(self, mode):
        self._metrics = TaTQAEmAndF1(mode=mode)

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, op_score = self._metrics.get_overall_metric(reset)
        print(f"raw matrix:{raw_detail}\r\n")
        print(f"detail em:{detail_em}\r\n")
        print(f"detail f1:{detail_f1}\r\n")
        print(f"global em:{exact_match}\r\n")
        print(f"global f1:{f1_score}\r\n")
        print(f"global scale:{scale_score}\r\n")
        print(f"global op:{op_score}\r\n")
        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail
