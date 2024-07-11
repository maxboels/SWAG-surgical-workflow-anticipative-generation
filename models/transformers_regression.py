import torch
import torch.nn as nn
import torch.nn.functional as F






class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  # Output layer to map to future embeddings

    def forward(self, queries, memory, current_pred=None, current_gt=None):
        tgt_embedded = self.embedding(queries) + self.positional_encoding[:, :queries.size(1), :]
        memory_embedded = self.embedding(memory) + self.positional_encoding[:, :memory.size(1), :]

        output = self.transformer_decoder(tgt_embedded.transpose(0, 1), memory_embedded.transpose(0, 1))
        output = self.output_layer(output.transpose(0, 1))

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        self.pe = self._compute_pe(seq_len)
        self.pe = self.pe.to(x.device)  # move self.pe to the same device as x
        return x + self.pe[:seq_len, :]

    def _compute_pe(self, length):
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, length, d_model)
        return pe
