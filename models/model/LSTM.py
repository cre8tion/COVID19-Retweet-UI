import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
from numpy import array

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GenModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        seq_length,
        n_layers,
        hidden_layers,
        bidirectional,
        dropout=0.5,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            856,
            hidden_dim[0],
            num_layers=n_layers,  # set to two: makes our LSTM 'deep'
            bidirectional=bidirectional,  # bidirectional or not
            dropout=dropout,
            batch_first=True,
        )  # we add dropout for regularization

        if bidirectional:
            self.D = 2
        else:
            self.D = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim[0]
        self.nonlinearity = nn.ReLU()
        self.hidden_layers = nn.ModuleList([])
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        assert len(hidden_dim) > 0
        assert len(hidden_dim) == 1 + hidden_layers

        i = 0
        if hidden_layers > 0:
            self.hidden_layers.append(
                nn.Linear(hidden_dim[i] * self.D * self.seq_length, hidden_dim[i + 1])
            )
            for i in range(hidden_layers - 1):
                self.hidden_layers.append(
                    nn.Linear(hidden_dim[i + 1], hidden_dim[i + 2])
                )
            self.output_projection = nn.Linear(hidden_dim[i + 1], 1)
        else:
            self.output_projection = nn.Linear(
                hidden_dim[i] * self.D * self.seq_length, 1
            )

    def forward(self, x, hidden):
        batch_size = x.size(0)
        val, hidden = self.rnn(x, hidden)  # feed to rnn

        # unpack sequence
        val = val.contiguous().view(batch_size, -1)
        for hidden_layer in self.hidden_layers:
            val = hidden_layer(val)
            val = self.dropout(val)
            val = self.nonlinearity(val)
        out = self.output_projection(val)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers * self.D, batch_size, self.hidden_dim)
            .zero_()
            .to(device),
            weight.new(self.n_layers * self.D, batch_size, self.hidden_dim)
            .zero_()
            .to(device),
        )

        return hidden


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def load_model(file):
    newmodel = GenModel([256], 30, 2, 0, True, 0.5).double()
    newmodel.to(device)
    newmodel.load_state_dict(torch.load(file, map_location=device))
    newmodel.eval()

    return newmodel


def load_prediction(df):
    model = load_model("./models/model/LSTM.pt")
    df2 = df[[c for c in df if c not in ["Retweets"]] + ["Retweets"]]
    n_timesteps = 30
    batch_size = 71  # tweak for number of files you want
    x1, y1 = split_sequences(df2.fillna(0).to_numpy(), n_timesteps)  # for the whole df
    x1 = torch.Tensor(x1).double().to(device)
    h = model.init_hidden(batch_size)
    hcon = tuple([e.data for e in h])
    predictions = model(x1, hcon)
    predicted_retweets = predictions[0].view(
        -1,
    )
    return predicted_retweets


if __name__ == "__main__":
    df = pd.read_feather("../../data/data_188489.ftr")
    load_prediction(df)
