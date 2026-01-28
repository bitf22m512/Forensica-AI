# ============================================
# RNN CLASSIFIER (LSTM / GRU) – Hybrid Ready
# ============================================
import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    """
    LSTM-based video classifier
    Input: [B, T, feature_dim]
    Output: [B, rnn_hidden_dim] (can feed into MCTNN or FC)
    """
    def __init__(self,
                 feature_dim=256,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=False,
                 rnn_type="LSTM",
                 dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

        self.output_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        """
        x: [B, T, feature_dim]
        returns: [B, rnn_output_dim]
        """
        out, _ = self.rnn(x)

        # Take last timestep
        if self.bidirectional:
            last_hidden = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        else:
            last_hidden = out[:, -1, :]

        return last_hidden
