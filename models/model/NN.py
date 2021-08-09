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


class MLP(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_dim, dropout):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList([])
        self.hidden_layers.append(nn.Linear(input_size, hidden_dim))
        for i in range(num_hidden - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, 1)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.dropout(x)
            x = self.nonlinearity(x)
        out = self.output_projection(x)
        return out


def split_sequences(sequences):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + 1
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def load_model(file):
    newmodel = MLP(856, 3, 256, 0.5).double()
    newmodel.to(device)

    newmodel.load_state_dict(torch.load(file, map_location=device))
    newmodel.eval()

    return newmodel


def load_nn_prediction(df):
    model = load_model("./models/model/NN.pt")

    df2 = df[[c for c in df if c not in ["Retweets"]] + ["Retweets"]]
    batch_size = 100  # tweak for number of files you want
    x1, y1 = split_sequences(df2.fillna(0).to_numpy())  # for the whole df
    # print(x1)
    x1 = torch.Tensor(x1).double().to(device)
    predictions = model(x1)
    predicted_retweets = predictions.view(
        -1,
    )

    return predicted_retweets


if __name__ == "__main__":
    df = pd.read_feather("../../data/data_188489.ftr")
    print(load_nn_prediction(df))
    # modelload((856, 3, 256, 0.5),'./NN.pt','../../data/data_188489.ftr')
