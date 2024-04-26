import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra.types import TargetConf
from omegaconf import OmegaConf
from . import key_recorder as kr
from . import transformer as tr
import time
import math
from typing import List

def corr_function(pred_values, eos_length=30*60):
    """ correct the predicted values by decrementing the values based on the time to the end of the sequence """
    corrected_values = []
    seq_length = len(pred_values)
    for i in range(seq_length):
        decrement = (((seq_length - i)*60) / seq_length) * (1/eos_length)
        corrected_value = pred_values[i] - decrement
        corrected_values.append(corrected_value)
    return corrected_values

class TokenPoolerCumMax(nn.Module):
    def __init__(self, dim: int = 512, pooling_dim: int = 64, frames_per_token: int = 60, 
                 relu_norm: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.pooling_dim = pooling_dim
        self.frames_per_token = frames_per_token
        
        if relu_norm:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, pooling_dim),
                nn.ReLU(),
                nn.LayerNorm(pooling_dim) # replace LayerNorm with Sonething for the Temporality
            )
        else:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, pooling_dim),
                nn.Sigmoid()
            )
        
        self.linear_expand = nn.Sequential(
            nn.Linear(pooling_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        num_ctx_tokens = seq_len // self.frames_per_token
        x = x.view(batch_size, num_ctx_tokens, self.frames_per_token, self.dim)
        x = self.linear_reduction(x)
        x, _ = torch.max(x, dim=2)
        x = self.linear_expand(x)
        return x

class TokenPoolerTopK(nn.Module):
    """ Association is All You Need """
    def __init__(self, dim: int = 512, pooling_dim: int = 64, top_k: int = 50, 
                 relu_norm: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.pooling_dim = pooling_dim
        self.top_k = top_k

        if relu_norm:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, pooling_dim),
                nn.ReLU(),
                nn.LayerNorm(pooling_dim) # replace LayerNorm with Sonething for the Temporality
            )
        else:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, pooling_dim),
                nn.Sigmoid()
            )
        
        self.linear_expand = nn.Sequential(
            nn.Linear(pooling_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim))
    
    def process_sequence(self, seq, k):
        summed_seq = torch.sum(sigmoid_seq, dim=0)
        top_k_values, top_k_indices = torch.topk(summed_seq, k)
        sampled_seq = seq[:, top_k_indices]
        return sampled_seq
    
    def forward(self, x):        
        x = self.linear_reduction(x)
        x = self.process_sequence(x, self.top_k)
        x = self.linear_expand(x)
        return x
        
class KeyRecorder(nn.Module):
    def __init__(self,
                dim: int = 512, 
                reduc_dim: int = 64, 
                sampling_rate: int = 10, 
                local_size: int = 20,
                return_max_now_reduc: bool = False,
                relu_norm: bool = True,
                **kwargs) -> None:
        super().__init__()
        self.return_max_now_reduc = return_max_now_reduc

        if relu_norm:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, reduc_dim),
                nn.ReLU(),
                nn.LayerNorm(reduc_dim) # replace LayerNorm with Sonething for the Temporality
            )
        else:
            self.linear_reduction = nn.Sequential(
                nn.Linear(dim, reduc_dim),
                nn.Sigmoid()
            )
        
        self._linear_expand = nn.Sequential(
            nn.Linear(reduc_dim, dim), 
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self._sampling_rate = sampling_rate
        self._local_size = local_size
        self.dropout = nn.Dropout(0.1)
        
    def compare_past_present(self, past_pooled_reduc: torch.Tensor, past_present: torch.Tensor) -> torch.Tensor:
        present = past_present[:, -self._local_size:, :]
        global_max = 0 + present
        for i in range(present.size(1)):
            past_pooled_reduc = torch.max(past_pooled_reduc, present[:,i,:])
            global_max[:,i,:] = past_pooled_reduc
        return global_max, past_pooled_reduc.unsqueeze(1)
        
    def forward(self, obs_frames, future=None) -> torch.Tensor:

        if future is not None:
            obs_frames = torch.cat([obs_frames, future], dim=1)

        compressed_feats = self.linear_reduction(obs_frames)
        past_pooled_reduc = torch.max(compressed_feats[:, :-self._local_size :self._sampling_rate,:], dim=1)[0]
        global_reduc_max, now_reduc_max = self.compare_past_present(past_pooled_reduc, compressed_feats)

        global_max = self._linear_expand(global_reduc_max)
        # global_max = self.dropout(global_max)
        
        # comment out option
        now_reduc_max = compressed_feats[:, -1:, :] # no pooling, just the last frame
        
        if self.return_max_now_reduc:
            return global_max, now_reduc_max
        
        return global_max, None

class TransformerEncoder(nn.Module):
    def __init__(self, input_length=20, input_dim=768, d_model=512, n_heads=8, num_layers=2,
                 dim_ff=2048, dropout=0.1,
                 activation='relu',
                 reshape_output=False):
        
        """From Memory-Anticipation-Transformer paper."""
        super().__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.d_model = d_model
        self.reshape_output = reshape_output

        self.data_embedding = tr.DataEmbedding(input_dim, d_model, dropout)

        encoder_layer = tr.TransformerEncoderLayer(d_model, n_heads, dim_ff, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = tr.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        B, T, D = src.shape
        src = src.contiguous().view(-1, self.input_length, self.input_dim)
        src = self.data_embedding(src).permute(1, 0, 2) # (B, T, D) -> (T, B, D)
        out = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2) # (T, B, D) -> (B, T, D)

        if self.reshape_output:
            return out.contiguous().view(B, T, self.d_model)

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim=512, d_model=512, n_heads=8,
                 num_layers=1, dim_ff=2048, dropout=0.1,
                 activation='relu'):
        
        """From Memory-Anticipation-Transformer paper."""
        super().__init__()
        self.d_model = d_model

        self.pos_encoding = tr.PositionalEncoding(d_model, dropout)
        self.data_embedding = tr.DataEmbedding(input_dim, d_model, dropout)

        decoder_layer = tr.TransformerDecoderLayer(d_model, n_heads, dim_ff, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = tr.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        
    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, knn=False):
        
        tgt = self.pos_encoding(tgt).permute(1, 0, 2) # (B, T, D) -> (T, B, D)
        memory = self.pos_encoding(memory).permute(1, 0, 2) # (B, T, D) -> (T, B, D)
        print(f"[TransformerDecoder] tgt (pe): {tgt.shape}")
        print(f"[TransformerDecoder] memory (pe): {memory.shape}")

        out = self.decoder(tgt,
                           memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           knn=knn)
        out = out.permute(1, 0, 2) # (T, B, D) -> (B, T, D)

        return out

class FusionHead(nn.Module):
    def __init__(
        self, 
        dim: int = 512, 
        **kwargs
    ):
        super().__init__()
        self.use_dropout = False
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, key_feats, feats):
        feats = self.norm1(key_feats + feats)
        if self.use_dropout:
            feats = self.norm2(self.dropout(self.fc_layer(feats)) + feats)
        else:
            feats = self.norm2(self.fc_layer(feats) + feats)
        return feats

class FusionHeadSegment(nn.Module):
    def __init__(
        self, 
        dim: int = 512, 
        **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_layer = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.2)


    def forward(self, key_feats, feats):
        feats = self.norm1(key_feats[:, -feats.shape[1]:, :] + feats)
        feats = self.norm2(self.dropout(self.fc_layer(feats)) + feats)
        return feats

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from hydra.types import TargetConf
from omegaconf import OmegaConf
from transformers import GPT2Model, GPT2Config

# local imports
# from . import key_recorder as kr
from . import transformer as tr


class R2A2(nn.Module):
    """ Reflective and Anticipative Action Recognition Model (R2A2) """
    def __init__(
        self,
        decoder_type: str = "ar_causal",
        eos_regression: bool = False,
        input_dim: int = 768,
        present_length: int = 20,
        num_ant_queries: int = 20,
        past_sampling_rate: int = 10,
        d_model: int = 512,
        num_classes: int = 7,
        max_future_preds: int = 30,
        frames_per_token: int = 60,
        pooling_dim: int = 64,
        gpt_cfg: dict = None,
        key_recorder: TargetConf = None,
        fusion_head: TargetConf = None,
        encoder: TargetConf = None,
        decoder: TargetConf = None,
        informer: TargetConf = None,
        **kwargs
    ):
        super().__init__()
        self.decoder_type = decoder_type
        self.input_dim = input_dim
        self.d_model = d_model
        self.present_length = present_length
        self.past_sampling_rate = past_sampling_rate
        self.max_future_preds = max_future_preds     # 60 if 30 frames per tokens to get 30 minutes
        self.frames_per_token = frames_per_token


        # other params
        self.relu_norm = True
        self.pooling_method = "cum_max"         # compare with baseline: "key_recorder"
        self.frame_level = True
        self.segment_level = False
        self.multiscale = False
        self.use_key_recorder = True
        self.decoder_type = "ar_causal"
        self.fixed_ctx_length = False
        # -----------------------------------------------------------------------

        self.proj_layer = nn.Linear(d_model, gpt_cfg.n_embd)

        # self.future_durations = nn.Sequential(
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.LayerNorm(64),
        #     nn.Linear(64, 1),
        # )
        
        # self.norm1 = nn.LayerNorm(d_model)

        self.informer = hydra.utils.instantiate(informer, _recursive_=False)
        # self.key_recorder = hydra.utils.instantiate(key_recorder, _recursive_=False)
        self.fusion_head1 = hydra.utils.instantiate(fusion_head, _recursive_=False)

        if self.pooling_method == "cum_max":
            self.tokens_pooler = TokenPoolerCumMax(dim=d_model, pooling_dim=pooling_dim, frames_per_token=self.frames_per_token,
                                                   relu_norm=self.relu_norm)
        elif self.pooling_method == "top_k":
            self.tokens_pooler = TokenPoolerTopK(dim=d_model, pooling_dim=pooling_dim, top_k=50, relu_norm=self.relu_norm)
        elif self.pooling_method == "key_recorder":
            self.tokens_pooler = KeyRecorder(dim=d_model, reduc_dim=pooling_dim, sampling_rate=10, local_size=20,
                                             relu_norm=self.relu_norm)
        else:
            raise ValueError(f"Pooling method {self.pooling_method} not implemented.")

        # Current Token and EOS Time prediction
        self.curr_frames_classifier = nn.Linear(d_model, num_classes)

        self.eos_regression = eos_regression
        if self.eos_regression:
            self.curr_eos_rem_time_reg = nn.Linear(d_model, 1)
        
        # Next Token Prediction
        self.next_action_classifier = nn.Linear(gpt_cfg.n_embd, num_classes) # optional +1 for the EOS (end of sequence token)

        # Phases
        if self.segment_level or self.multiscale:
            self.phase_proj = nn.Linear(d_model, 128)
            self.next_phase_classifier = nn.Linear(128, num_classes)

        if self.decoder_type == "ar_causal":
            if self.frame_level or self.multiscale:
                # TODO: Ablations change vocab size 
                config = GPT2Config(vocab_size=gpt_cfg.vocab_size,          # number of different tokens (classes) that can be represented by the inputs_ids
                                    n_positions=256,                        # number of positional embeddings
                                    n_embd=gpt_cfg.n_embd,                  # embedding dimension
                                    n_layer=gpt_cfg.n_layer,                # try l:8 / m:6 / s:4 / xs:2
                                    n_head=gpt_cfg.n_head,                  # same
                                    use_cache=True)                         # whether to use cache for the model
                self.frame_decoder = GPT2Model(config)
            
            if self.segment_level or self.multiscale:
                # Phase Decoder
                config_phase = GPT2Config(vocab_size=100, 
                                        n_positions=256, 
                                        n_embd=128, 
                                        n_layer=gpt_cfg.n_layer, 
                                        n_head=8, 
                                        use_cache=True)
                self.long_term_decoder = GPT2Model(config_phase)
        else:
            self.frame_decoder = tr.TransformerDecoder(**decoder)
            self.long_term_decoder = tr.TransformerDecoder(**decoder)

    def encoder(self, obs_video):
        """Follow the skit implementation."""
        B, _, _ = obs_video.shape
        past = obs_video[:, :-self.present_length]
        present = obs_video[:, -self.present_length:]
        past = past.contiguous().view(-1, self.present_length, self.input_dim)
        past = self.informer(past, past)
        past = past.contiguous().view(B, -1, self.d_model)
        present = self.informer(present, present)
        past_present = torch.cat((past, present), dim=1)
        return past_present
    
    def forward(self, obs_video, train_mode=True,
                future_labels=None, pad_mask=None,
                level="frame", multiscale=False, lt_pooling="token_pooling"):
        print(f"[R2A2] obs_video: {obs_video.shape}")
        outputs = {}
        enc_out = self.encoder(obs_video)  # sliding window encoder
        print(f"[R2A2] enc_out: {enc_out.shape}")

        # TODO: ABLATIONS FOR FRAME AND SEGMENT DECODING
        if lt_pooling == "key_recorder":
            enc_out_pooled, now_reduc_max = self.tokens_pooler(enc_out)
        elif lt_pooling == "token_pooling":
            enc_out_pooled = self.tokens_pooler(enc_out)
        print(f"[R2A2] enc_out_pooled: {enc_out_pooled.shape}")

        # Local Context skip connection fusion
        if lt_pooling == "key_recorder":
            enc_out_local = enc_out[:, -self.present_length:]
        elif lt_pooling == "token_pooling":
            enc_out_local = enc_out[:, self.frames_per_token-1:enc_out.size(1):self.frames_per_token, :]
        print(f"[R2A2] enc_out_local: {enc_out_local.shape}")
        
        # Fusion Head (skip connection + linear layer)
        assert enc_out_pooled.size(1) == enc_out_local.size(1)

        enc_out = self.fusion_head1(enc_out_pooled, enc_out_local)
        print(f"[R2A2] enc_out_fused: {enc_out.shape}")


        curr_frames_cls = self.curr_frames_classifier(enc_out)
        print(f"[R2A2] curr_frames: {curr_frames_cls.shape}")
        outputs["curr_frames"] = curr_frames_cls

        # Current EOS Time Prediction
        if self.eos_regression:
            curr_eos_rem_time = self.curr_eos_rem_time_reg(enc_out)
            print(f"[R2A2] curr_eos_rem_time: {curr_eos_rem_time.shape}")
            outputs["curr_eos_rem_time"] = curr_eos_rem_time


        if train_mode:
            # GPT-2 decoder takes a single sequence as input prompt and predicts the next token.
            # In this setup, the next token is the next action in the sequence at a frame level since
            # the input context is also at a frame level.
            # However, the long context sequence has been pooled over key-features and added to the local context window.
            # So the next token representation should include the long-term context and the next short-term context.
            # A single token is represented by all the phases activations from previous frames.
            # So it should predict if new phase features activations are going to appear in the next frames.

            if level == "frame" or multiscale:
                dec_in = self.proj_layer(enc_out)
                next_action = self.frame_decoder(inputs_embeds=dec_in)  # next frame-level prediction
                next_frames_cls = self.next_action_classifier(next_action.last_hidden_state)
                outputs["next_frames"] = next_frames_cls

            if level == "segment" or multiscale:
                # Long-Term Planning
                phase_embed = self.phase_proj(enc_out)
                next_segmts = self.long_term_decoder(inputs_embeds=phase_embed)    # next phase-level prediction
                next_segmts_cls = self.next_phase_classifier(next_segmts.last_hidden_state)
                outputs["next_segmts"] = next_segmts_cls
        else:
            # Autoregressive decoding during inference
            # TODO: TRY WITH FIXED CONTEXT WINDOW TO PREVENT LATENCY DURING INFERENCE
            
            if "curr_eos_rem_time" in outputs:
                curr_eos_rem_time = outputs["curr_eos_rem_time"]

                # TODO: ABLATIONS FOR EOS INFERENCE USAGE
                # add the distance to current frame to every indices
                # corr_eos_rem_time = corr_function(curr_eos_rem_time.squeeze().tolist())

                # mean over batch for more stable estimation (1fps)
                mean_eos_estim_time = curr_eos_rem_time.mean().item()

                num_future_predictions = self.max_future_preds * mean_eos_estim_time
                print(f"[R2A2] num_future_predictions: {num_future_predictions}")
            else:
                num_future_predictions = self.max_future_preds
            
            iters_time = []
            frames_cls_preds = []
            dec_in = self.proj_layer(enc_out)
            for _ in range(num_future_predictions):
                start_time = time.time()
                next_frames = self.frame_decoder(inputs_embeds=dec_in)
                iters_time.append(time.time() - start_time) # measure time per iteration
                next_frame_embed = next_frames.last_hidden_state[:, -1:, :]
                next_frame_cls = self.next_action_classifier(next_frame_embed)
                frames_cls_preds.append(next_frame_cls)
                # shift the input sequence by one frame if fixed context length
                if self.fixed_ctx_length:
                    dec_in = dec_in[:, 1:]
                dec_in = torch.cat((dec_in, next_frame_embed), dim=1)
                print(f"[R2A2] dec_in: {dec_in.shape}")
            outputs["future_frames"] = torch.cat(frames_cls_preds, dim=1)
            outputs["iters_time"] = iters_time
            print(f"[R2A2] future_frames: {outputs['future_frames'].shape}")
            print(f"[R2A2] iters_time: {outputs['iters_time']}")

        return outputs

        