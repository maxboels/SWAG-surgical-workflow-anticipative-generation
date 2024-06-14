import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # self.pos_encoding = nn.init.normal_(torch.zeros(1, 1000, hidden_dim), mean=0.0, std=0.02)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000
        # self.positional_encoding.data.normal_(mean=0.0, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  # Output layer to map to future embeddings

    def forward(self, tgt, memory):
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        memory_embedded = self.embedding(memory) + self.positional_encoding[:, :memory.size(1), :]

        output = self.transformer_decoder(tgt_embedded.transpose(0, 1), memory_embedded.transpose(0, 1))
        output = self.output_layer(output.transpose(0, 1))

        return output