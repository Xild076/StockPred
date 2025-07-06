import torch
import torch.nn as nn
import math

class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, ticker_count, d_model, n_heads, n_layers, dropout, ticker_embedding_dim, sequence_length, prediction_horizon=3):
        super(SimpleTransformerModel, self).__init__()
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        self.ticker_embed = nn.Embedding(ticker_count, ticker_embedding_dim)
        
        total_input_dim = self.input_dim + ticker_embedding_dim
        self.input_projection = nn.Linear(total_input_dim, d_model)
        
        self.pos_embedding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, prediction_horizon * self.input_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, ticker_ids):
        batch_size, seq_len, _ = src.shape
        
        ticker_emb = self.ticker_embed(ticker_ids)
        ticker_emb = ticker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        x = torch.cat([src, ticker_emb], dim=-1)
        x = self.input_projection(x)
        
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_emb
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1)
        
        output = self.output_projection(x)
        output = output.view(batch_size, self.prediction_horizon, self.input_dim)
        
        return output

UniversalTransformerModel = SimpleTransformerModel