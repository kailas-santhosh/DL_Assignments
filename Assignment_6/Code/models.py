import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, model_type):
        super().__init__()
        self.model_type = model_type
        dropout = 0.2 if num_layers > 1 else 0.0
        if model_type == "RNN":
            self.recurrent = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                nonlinearity="tanh", batch_first=True, dropout=dropout
            )
        else:
            self.recurrent = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.model_type == "RNN":
            _, hidden = self.recurrent(packed)
        else:
            _, (hidden, _) = self.recurrent(packed)
        return self.classifier(hidden[-1])
