# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
assert torch.__version__ >= '1.6.0'
import torch.nn as nn


def layer_norm(d_model, condition=True):
    return nn.LayerNorm(d_model) if condition else None


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # mask = mask.float().masked_fill(mask == 0, float(-1e5)).masked_fill(mask == 1, float(0.0))
    return mask

def generate_causal_attention_mask(max_seq_len):
    """
    Generates a causal attention mask with float -inf for a given batch size and maximum sequence length.
    """
    # Create a tensor of shape (batch_size, max_seq_len, max_seq_len) filled with float -inf
    mask = torch.full((max_seq_len, max_seq_len), float('-inf'))
    
    # Fill the diagonal and lower triangle with zeros
    mask = torch.triu(mask, diagonal=1)
    
    return mask