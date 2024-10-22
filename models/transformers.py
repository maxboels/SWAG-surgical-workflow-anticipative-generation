import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


class ClassConditionedTransformerDecoder(nn.Module):
    def __init__(self, cfg, num_queries, input_dim, hidden_dim, n_heads, n_layers, num_classes, 
                    conditional_probs_embeddings=True,
                    normalize_priors=False, dim_feedforward=2048, dropout=0.1):
        super(ClassConditionedTransformerDecoder, self).__init__()

        dataset = cfg.dataset
        h = cfg.horizon

        self.do_classification = cfg.do_classification
        self.do_regression = cfg.do_regression

        self.num_classes = num_classes
        self.conditional_probs_embeddings = conditional_probs_embeddings
        self.normalize_priors = normalize_priors

        if self.normalize_priors:
            self.norm_layer = nn.LayerNorm(num_classes)      
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.class_projection_layer = nn.Linear(num_classes, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.conditional_probs_embeddings:
            if self.do_classification:
                # num_classes = num_classes + 1  # Add EOS class
                root = "/nfs/home/mboels/projects/SuPRA/datasets"
                path_class_probs = root + f"/{dataset}/naive2_{dataset}_class_probs_at{h}.json"
                with open(path_class_probs, 'r') as f:
                    class_freq_pos = json.load(f)
                class_freq_pos = {int(k): [{int(inner_k): inner_v for inner_k, inner_v in freq_dict.items()} for freq_dict in v] for k, v in class_freq_pos.items()}
            elif self.do_regression and self.conditional_probs_embeddings:
                file_path = f"/nfs/home/mboels/projects/SuPRA/datasets/{dataset}/rem_time_{h}_conditional_probs.npy"
                if os.path.exists(file_path):
                    self.rt_conditional_probs = np.load(file_path)
                    self.rt_conditional_probs = torch.tensor(self.rt_conditional_probs).to(self.device)
                    print(f"[ClassConditionedTransformerDecoderRegression] rt_conditional_probs: {self.rt_conditional_probs.shape}")
                else:
                    self.conditional_probs_embeddings = False
                    print(f"[ClassConditionedTransformerDecoderRegression] File not found: {file_path}. Setting conditional_probs_embeddings to False.")
            else:
                raise ValueError("Multi-task learning not supported yet.")
        else:
            print("[ClassConditionedTransformerDecoder] No conditional probabilities used.")

    def forward(self, query_embeddings, memory, current_pred=None, current_gt=None):
        batch_size, seq_len, _ = query_embeddings.size()
        print(f"[ClassConditionedTransformerDecoder] query_embeddings: {query_embeddings.shape}")

        # Positional encoding for query_embeddings
        query_embeddings = self.positional_encoding(query_embeddings)
        print(f"[ClassConditionedTransformerDecoder] query_embeddings (with PE): {query_embeddings.shape}")

        # Embedding for memory 
        memory_embedded = self.embedding(memory) + self.positional_encoding(memory) # NOTE: no need for embedding layer
        print(f"[ClassConditionedTransformerDecoder] memory_embedded (with LN+PE): {memory_embedded.shape}")

        # Initialize combined embeddings
        input_embeddings = query_embeddings.clone().to(self.device)
        print(f"[ClassConditionedTransformerDecoder] input_embeddings (cloned query_embeddings): {input_embeddings.shape}")

        if self.conditional_probs_embeddings:
            # Initialize future class probabilities
            future_class_probs = torch.zeros(batch_size, seq_len, self.num_classes, device=self.device) + 1e-6
            print(f"[ClassConditionedTransformerDecoder] future_class_probs (init 1e-6): {future_class_probs.shape}")
            if self.do_classification:
                for i in range(batch_size):
                    for j in range(seq_len):
                        if current_gt is not None:
                            current_class = current_gt[i, -1].item()
                            # Use ground truth class for teacher forcing
                            class_probs = self.sampler.class_probs(current_class, j)
                            for k, v in class_probs.items():
                                future_class_probs[i, j, k] = v
                        elif current_pred is not None:
                            # Use predicted class for inference
                            current_class = torch.argmax(F.softmax(current_pred[i], dim=-1), dim=-1).item()
                            class_probs = self.sampler.class_probs(current_class, j)
                            for k, v in class_probs.items():
                                future_class_probs[i, j, k] = v
                        else:
                            raise ValueError("Either current_pred or current_gt must be provided.")
                print(f"[ClassConditionedTransformerDecoder] future classes position embedding: {future_class_probs.shape}")
            elif self.do_regression:
                for i in range(batch_size):
                    for j in range(seq_len):
                        if current_gt is not None:
                            # Use ground truth class for teacher forcing
                            current_class = current_gt[i, -1].item()
                            future_class_probs[i, j, :] = self.rt_conditional_probs[current_class, :]
                        elif current_pred is not None:
                            # Use predicted class for inference
                            current_class = torch.argmax(F.softmax(current_pred[i], dim=-1), dim=-1).item()
                            future_class_probs[i, j, :] = self.rt_conditional_probs[current_class, :]
                        else:
                            raise ValueError("Either current_pred or current_gt must be provided.")
                print(f"[ClassConditionedTransformerDecoder] remaining time embedding: {future_class_probs.shape}")
            else:
                raise ValueError("Multi-task learning not supported yet.")        
            # TODO: try to Normalize future_class_probs before passing to linear layer
            if self.normalize_priors:
                future_class_probs = self.norm_layer(future_class_probs)
            # Compute class-conditioned embeddings
            class_conditioned_embedding = self.class_projection_layer(future_class_probs)
            print(f"[ClassConditionedTransformerDecoder] class_conditioned_embedding (linear increase): {class_conditioned_embedding.shape}")
            input_embeddings += class_conditioned_embedding
            print(f"[ClassConditionedTransformerDecoder] input_embeddings (addition): {input_embeddings.shape}")
        else:
            print("[ClassConditionedTransformerDecoder] Skipping class-conditioned embeddings.")
        
        # Transformer decoder
        output = self.transformer_decoder(input_embeddings.transpose(0, 1), memory_embedded.transpose(0, 1))
        output = self.output_layer(output.transpose(0, 1))

        return output

class ClassConditionedTransformerDecoderRegression(nn.Module):
    def __init__(self, cfg, num_queries, input_dim, hidden_dim, n_heads, n_layers, num_classes, class_freq_positions, normalize_priors=False, dim_feedforward=2048, dropout=0.1):
        super(ClassConditionedTransformerDecoderRegression, self).__init__()

        dataset = cfg.dataset
        h = num_queries

        self.num_classes = num_classes # should be 8 with EOS class
        self.normalize_priors = normalize_priors

        if self.normalize_priors:
            self.norm_layer = nn.LayerNorm(num_classes)      
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.class_projection_layer = nn.Linear(num_classes, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # file_path = "datasets/cholec80/rem_time_18_conditional_probs.npy"
        file_path= f"/nfs/home/mboels/projects/SuPRA/datasets/{dataset}/rem_time_{h}_conditional_probs.npy"
        self.rt_conditional_probs = np.load(file_path)
        print(f"[ClassConditionedTransformerDecoderRegression] rt_conditional_probs: {self.rt_conditional_probs.shape}")

        self.rt_conditional_probs = torch.tensor(self.rt_conditional_probs).to(self.device)


    def forward(self, query_embeddings, memory, current_pred=None, current_gt=None):
        batch_size, seq_len, _ = query_embeddings.size()
        print(f"[ClassConditionedTransformerDecoderRegression] query_embeddings: {query_embeddings.shape}")

        # Positional encoding for query_embeddings
        query_embeddings = self.positional_encoding(query_embeddings)
        print(f"[ClassConditionedTransformerDecoderRegression] query_embeddings (with PE): {query_embeddings.shape}")

        # Embedding for memory 
        memory_embedded = self.embedding(memory) + self.positional_encoding(memory) # NOTE: no need for embedding layer
        print(f"[ClassConditionedTransformerDecoderRegression] memory_embedded (with LN+PE): {memory_embedded.shape}")

        # Initialize combined embeddings
        input_embeddings = query_embeddings.clone().to(self.device)
        print(f"[ClassConditionedTransformerDecoderRegression] input_embeddings (cloned query_embeddings): {input_embeddings.shape}")

        # Initialize future class probabilities
        future_class_probs = torch.zeros(batch_size, seq_len, self.num_classes, device=self.device) + 1e-6
        print(f"[ClassConditionedTransformerDecoderRegression] future_class_probs (init 1e-6): {future_class_probs.shape}")

        for i in range(batch_size):
            for j in range(seq_len):
                if current_gt is not None:
                    # Use ground truth class for teacher forcing
                    current_class = current_gt[i, -1].item()
                    future_class_probs[i, j, :] = self.rt_conditional_probs[current_class, :]
                elif current_pred is not None:
                    # Use predicted class for inference
                    current_class = torch.argmax(F.softmax(current_pred[i], dim=-1), dim=-1).item()
                    future_class_probs[i, j, :] = self.rt_conditional_probs[current_class, :]
                else:
                    raise ValueError("Either current_pred or current_gt must be provided.")
        
        print(f"[ClassConditionedTransformerDecoderRegression] future_class_probs (with conditional probs): {future_class_probs.shape}")
        print(f"[ClassConditionedTransformerDecoderRegression] future_class_probs (with conditional probs): {future_class_probs}")

        # TODO: try to Normalize future_class_probs before passing to linear layer
        if self.normalize_priors:
            future_class_probs = self.norm_layer(future_class_probs)

        # Compute class-conditioned embeddings
        class_conditioned_embedding = self.class_projection_layer(future_class_probs)
        print(f"[ClassConditionedTransformerDecoderRegression] class_conditioned_embedding (linear): {class_conditioned_embedding.shape}")
        input_embeddings += class_conditioned_embedding
        print(f"[ClassConditionedTransformerDecoderRegression] input_embeddings (addition): {input_embeddings.shape}")

        # Transformer decoder
        output = self.transformer_decoder(input_embeddings.transpose(0, 1), memory_embedded.transpose(0, 1))
        output = self.output_layer(output.transpose(0, 1))

        return output
