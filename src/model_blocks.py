import torch
import torch.nn as nn
import math

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, dropout=0.0):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context_vector)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2,
                 dropout=0.2, output_size=1):
        super(TCNModel, self).__init__()
        from torch.nn.utils import weight_norm
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            conv = weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=padding, dilation=dilation_size))
            layers += [conv, nn.ReLU(), nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        y = y[:, :, -1]
        y = self.fc(y)
        return y

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, num_layers, nhead,
                 hidden_size, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=nhead,
            dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size,
                 output_size, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=num_filters,
            kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.mean(dim=2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

MODEL_BLOCKS = {
    'lstm': LSTMAttentionModel,
    'gru': GRUModel,
    'tcn': TCNModel,
    'transformer': TimeSeriesTransformer,
    'cnn': CNNModel,
}