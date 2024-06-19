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

    def forward(self, queries, memory):
        tgt_embedded = self.embedding(queries) + self.positional_encoding[:, :queries.size(1), :]
        memory_embedded = self.embedding(memory) + self.positional_encoding[:, :memory.size(1), :]

        output = self.transformer_decoder(tgt_embedded.transpose(0, 1), memory_embedded.transpose(0, 1))
        output = self.output_layer(output.transpose(0, 1))

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import json

class GaussianMixtureSamplerWithPosition:
    def __init__(self, class_freq_positions, lookahead=18):
        self.class_freq_positions = class_freq_positions
        self.lookahead = lookahead
        self.probabilities = self._compute_probabilities()
    
    def _compute_probabilities(self):
        probabilities = {}
        for cls, freq_list in self.class_freq_positions.items():
            probabilities[int(cls)] = []
            for freq_dict in freq_list:
                total_count = sum(freq_dict.values())
                probabilities[int(cls)].append({int(k): v / total_count for k, v in freq_dict.items()})
        return probabilities
    
    def class_probs(self, current_class, position):
        if current_class not in self.probabilities:
            raise ValueError(f"Class {current_class} not found in class frequencies.")
        
        return self.probabilities[current_class][position]
    
    def sample_class(self, current_class, num_samples=18):
        if current_class not in self.probabilities:
            raise ValueError(f"Class {current_class} not found in class frequencies.")
        
        samples = []
        for j in range(num_samples):
            possible_values = list(self.probabilities[current_class][j].keys())
            probabilities = list(self.probabilities[current_class][j].values())
            samples.append(np.random.choice(possible_values, p=probabilities))
        
        return samples

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ClassConditionedTransformerDecoder(nn.Module):
    def __init__(self, num_queries, input_dim, hidden_dim, n_heads, n_layers, num_classes, class_freq_positions, dim_feedforward=2048, dropout=0.1):
        super(ClassConditionedTransformerDecoder, self).__init__()

        self.num_classes = num_classes

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.class_projection_layer = nn.Linear(num_classes, hidden_dim)
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000
        # self.positional_encoding = nn.Embedding(1000, hidden_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        self.query_embeddings = nn.Embedding(num_queries, embedding_dim)
        nn.init.xavier_uniform_(self.query_embeddings.weight)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  # Output layer to map to future embeddings
        self.sampler = GaussianMixtureSamplerWithPosition(class_freq_positions, lookahead=18)

        # Check if a GPU is available and if not, use a CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, queries, memory, current_pred=None, current_gt=None):
        batch_size, seq_len, _ = queries.size()

        # Embedding for queries with dimension (num_queries, hidden_dim)
        query_embeddings = self.query_embedding(queries)
        query_embeddings = self.positional_encoding(query_embeddings)
        
        # Adding positional encoding to the input embeddings
        tgt_embedded = self.embedding(queries) + self.positional_encoding[:, :seq_len, :]
        memory_embedded = self.embedding(memory) + self.positional_encoding[:, :memory.size(1), :]
        print(f"[Decoder CC] tgt_embedded: {tgt_embedded.size()}")
        print(f"[Decoder CC] memory_embedded: {memory_embedded.size()}")

        # Init combined embeddings with size (seq_len, batch_size, hidden_dim)
        combined_embeddings = torch.zeros_like(tgt_embedded).to(self.device)
        print(f"[Decoder CC] combined_embeddings (init): {combined_embeddings.size()}")

        # Init class probs vector with size (seq_len, num_classes)
        future_class_probs = torch.zeros(batch_size, seq_len, self.num_classes) + 1e-6
        future_class_probs = future_class_probs.to(self.device)
        print(f"[Decoder CC] future_class_probs (init): {future_class_probs.size()}")

        for i in range(batch_size):
            for j in range(seq_len):
                if current_gt is not None:
                    # Use ground truth class for teacher forcing
                    current_class = current_gt[i].item()
                    class_probs = self.sampler.class_probs(current_class, j)
                    for k, v in class_probs.items():
                        future_class_probs[i, j, k] = v
                elif current_pred is not None:
                    # Use predicted class for inference
                    print(f"[Decoder CC] current_pred: {current_pred.size()}")
                    current_class = torch.argmax(F.softmax(current_pred[i], dim=-1), dim=-1).item()
                    class_probs = self.sampler.class_probs(current_class, j)
                    for k, v in class_probs.items():
                        future_class_probs[i, j, k] = v
                else:
                    raise ValueError("Either current_pred or current_gt must be provided.")
                
        # Compute class conditioned embeddings in batch
        class_conditioned_embedding = self.class_projection_layer(future_class_probs)
        combined_embeddings = tgt_embedded + class_conditioned_embedding
        print(f"[Decoder CC] combined_embeddings: {combined_embeddings.size()}")

        output = self.transformer_decoder(combined_embeddings.transpose(0, 1), memory_embedded.transpose(0, 1))
        print(f"[Decoder CC] output (transformer): {output.size()}")
        output = self.output_layer(output.transpose(0, 1))
        print(f"[Decoder CC] output (linear): {output.size()}")

        return output