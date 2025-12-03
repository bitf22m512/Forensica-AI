import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    LSTM-based video classifier.
    Input:  [B, T, feature_dim]
    Output: [B, 2] (real/fake logits)
    """

    def __init__(self,
                 feature_dim=256,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.3):
        super(RNNClassifier, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

        # Output dimension
        rnn_output_dim = hidden_size * (2 if bidirectional else 1)

        # Final classifier
        self.fc = nn.Linear(rnn_output_dim, 2)  # 2 classes: real/fake

    def forward(self, x):
        """
        x shape: [B, T, feature_dim]

        Returns: logits [B, 2]
        """

        # LSTM outputs:
        #   out:      [B, T, hidden]
        #   (h_n, c_n)
        out, (h_n, c_n) = self.lstm(x)

        # We use the last hidden state of the final LSTM layer
        if self.bidirectional:
            # concatenate last forward and backward hidden states
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, hidden*2]
        else:
            last_hidden = h_n[-1]  # [B, hidden]

        logits = self.fc(last_hidden)  # [B, 2]

        return logits
