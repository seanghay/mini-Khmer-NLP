import torch.nn as nn

class KccRNN(nn.Module):
    def __init__(
        self,
        tokens,
        embedding_dim=50,
        n_hidden=256,
        n_layers=2,
        drop_prob=0.5,
        lr=0.001,
    ):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.kccs = tokens
        self.int2kcc = dict(enumerate(self.kccs))
        self.kccs2int = {ch: ii for ii, ch in self.int2kcc.items()}
        self.word_embeddings = nn.Embedding(len(self.kccs), embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            n_hidden,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden * 2, 1)

    def forward(self, x, hidden):
        """Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`."""
        embeds = self.word_embeddings(x)
        r_output, hidden = self.lstm(embeds, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden * 2)
        out = self.fc(out)
        return out.squeeze(), hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_(),
        )
        return hidden
