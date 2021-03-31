import torch.nn as nn
import torch
from tag_op.tagop.newmo_util import Default_FNN
from tag_op.tagop.tools.allennlp import replace_masked_values, masked_log_softmax



class SequenceTagHead(nn.Module):

    def __init__(self, hidden_size, dropout_prob):
        super(SequenceTagHead, self).__init__()
        self.tag_predictor = Default_FNN(hidden_size, hidden_size, 2, dropout_prob)
        self.NLLLoss = nn.NLLLoss(reduction="sum")

    def forward(self, input_vec: torch.LongTensor, table_mask: torch.LongTensor, paragraph_mask: torch.LongTensor,labels: torch.LongTensor):
        table_sequence_output = replace_masked_values(input_vec, table_mask.unsqueeze(-1), 0)
        table_tag_prediction = self.tag_predictor(table_sequence_output)
        table_tag_prediction = masked_log_softmax(table_tag_prediction, mask=None)
        table_tag_prediction = replace_masked_values(table_tag_prediction, table_mask.unsqueeze(-1), 0)
        table_tag_labels = replace_masked_values(labels.float(), table_mask, 0)

        paragraph_sequence_output = replace_masked_values(input_vec, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_prediction = self.tag_predictor(paragraph_sequence_output)
        paragraph_tag_prediction = masked_log_softmax(paragraph_tag_prediction, mask=None)
        paragraph_tag_prediction = replace_masked_values(paragraph_tag_prediction, paragraph_mask.unsqueeze(-1), 0)
        paragraph_tag_labels = replace_masked_values(labels.float(), paragraph_mask, 0)

        table_tag_prediction = table_tag_prediction.transpose(1, 2)  # [bsz, 2, table_size]
        table_tag_prediction_loss = self.NLLLoss(table_tag_prediction, table_tag_labels.long())
        paragraph_tag_prediction = paragraph_tag_prediction.transpose(1, 2)
        paragraph_token_tag_prediction_loss = self.NLLLoss(paragraph_tag_prediction, paragraph_tag_labels.long())

        loss = table_tag_prediction_loss + paragraph_token_tag_prediction_loss

        return loss

