import math
import torch
import torch.nn as nn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Default_FNN(nn.Module):
    def __init__(self, input_size, mid_size, output_size, dropout, activation_fn=None, layer_norm=True):
        super(Default_FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, mid_size)
        if layer_norm:
            self.ln = nn.LayerNorm(mid_size)
        else:
            self.ln = None
        if activation_fn:
            self.afn = activation_fn
        else:
            self.afn = gelu
        self.dropout_fn = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mid_size, output_size)

    def forward(self, input: torch.LongTensor):
        out = self.afn(self.fc1(self.dropout_fn(input)))
        if self.ln:
            out = self.ln(out)
        return self.fc2(out)



