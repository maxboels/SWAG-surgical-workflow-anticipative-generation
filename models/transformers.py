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

    def forward(self, tgt, memory):
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
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

class ClassConditionedTransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, num_classes, class_freq_positions, dim_feedforward=2048, dropout=0.1):
        super(ClassConditionedTransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.class_projection_layer = nn.Linear(num_classes, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # Assuming max sequence length of 1000
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)  # Output layer to map to future embeddings
        self.sampler = GaussianMixtureSamplerWithPosition(class_freq_positions, lookahead=18)

    def forward(self, tgt, memory, current_pred=None, current_gt=None):
        batch_size, seq_len, _ = tgt.size()
        
        # Adding positional encoding to the input embeddings
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:, :seq_len, :]
        memory_embedded = self.embedding(memory) + self.positional_encoding[:, :memory.size(1), :]

        # Init combined embeddings with size (seq_len, batch_size, hidden_dim)
        combined_embeddings = torch.zeros_like(tgt_embedded)

        # Init class probs vector with size (seq_len, num_classes)
        future_class_probs = torch.zeros(batch_size, seq_len, num_classes) + 1e-6
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