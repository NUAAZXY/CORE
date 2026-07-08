# coding=utf-8
# COREGEN model adapted for Qwen2 architecture (transformers 4.44.2)
# Based on COREGEN7b.py (LLaMA) and transformers/models/qwen2/modeling_qwen2.py
#
# Key changes from standard Qwen2:
#   - DependencyEncoding integrated into attention layers (masking_layer)
#   - CodeDynamicCache for inference with dependency encoding state
#   - Block mask attention with dynamic TopK sparse activation
#   - Dual RoPE (standard + window) for long-context extrapolation

import math
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


# ============================================================
# CodeDynamicCache - extended cache for DependencyEncoding state
# ============================================================

class CodeDynamicCache(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.pos_embeddings: List[torch.Tensor] = [1, 1]
        self._seen_tokens = 0
        self.tokens = None
        self.de_k = []
        self.lde = None
        self.lbm = None
        self.f = []

    def __getitem__(self, layer_idx: int):
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update_de(self, q, k, i, split):
        if len(self.de_k) < len(split):
            self.de_k.append(k)
        else:
            self.de_k[i] = torch.cat([self.de_k[i], k], dim=-2)
        return q, self.de_k[i]

    def update_f(self, f, i, split):
        if len(self.f) < len(split):
            self.f.append(f)
        else:
            if f.squeeze(1).equal(self.f[i][:, -1, :]):
                self.f[i] = torch.cat([self.f[i], f], dim=-2)
            else:
                self.f[i] = f

    def find_position_difference(self, n):
        tensor = self.tokens[0]
        positions = torch.nonzero(tensor == 13).squeeze()
        if positions.dim() == 0:
            positions = positions.unsqueeze(0)
        if n > len(positions):
            target_position = positions[0] if len(positions) > 0 else 0
        else:
            target_position = positions[-n]
        position_difference = tensor.size(-1) - 1 - target_position
        return position_difference

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self):
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]


# ============================================================
# Helper functions
# ============================================================

def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask, sequence_length, target_length, dtype, device, min_dtype, cache_position, batch_size):
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
    return causal_mask


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device='cuda')
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


def dynamic_topk_attention(d_score, training_length=1024, max_k=512, is_prefilling=True, block_size=64):
    batch, head, query_len, seq_len = d_score.shape
    device = d_score.device
    final_mask = torch.ones_like(d_score, dtype=torch.bool)

    def get_k_for_position(pos):
        visible_tokens = pos + 1
        if pos < training_length:
            return min(visible_tokens, max_k)
        else:
            position_ratio = int(max_k * ((pos / training_length)))
            return min(visible_tokens, position_ratio)

    if is_prefilling:
        for block_start in range(0, query_len, block_size):
            block_end = min(block_start + block_size, query_len)
            for q_idx in range(block_start, block_end):
                visible_tokens = q_idx + 1
                k = get_k_for_position(q_idx)
                scores = d_score[:, :, q_idx, :visible_tokens]
                _, topk_indices = torch.topk(scores, k=k, dim=-1)
                batch_indices = torch.arange(batch, device=device).view(-1, 1, 1).expand(-1, head, k)
                head_indices = torch.arange(head, device=device).view(1, -1, 1).expand(batch, -1, k)
                q_indices = torch.full((batch, head, k), q_idx, device=device)
                final_mask[batch_indices, head_indices, q_indices, topk_indices] = False
    else:
        q_idx = query_len - 1
        visible_tokens = seq_len
        pos = visible_tokens - 1
        k = get_k_for_position(pos)
        scores = d_score[:, :, q_idx, :visible_tokens]
        _, topk_indices = torch.topk(scores, k=k, dim=-1)
        batch_indices = torch.arange(batch, device=device).view(-1, 1, 1).expand(-1, head, k)
        head_indices = torch.arange(head, device=device).view(1, -1, 1).expand(batch, -1, k)
        q_indices = torch.full((batch, head, k), q_idx, device=device)
        final_mask[batch_indices, head_indices, q_indices, topk_indices] = False

    if is_prefilling:
        causal_mask = torch.triu(torch.ones(query_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    else:
        causal_mask = torch.zeros(query_len, seq_len, device=device, dtype=torch.bool)
    final_mask = final_mask | causal_mask.unsqueeze(0).unsqueeze(0)
    return final_mask


# ============================================================
# DependencyEncoding - multi-level dictionary with TopK sparse activation
# ============================================================

class DependencyEncoding(nn.Module):
    def __init__(self, act_size, dict_size, block_indices, top_k):
        super().__init__()
        self.act_size = act_size
        self.dict_size = dict_size

        self.b_dec = nn.Parameter(torch.zeros(self.act_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.act_size, self.dict_size))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.dict_size, self.act_size))
        )
        self.W_dec.data[:] = self.W_enc.t().data
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        self.num_batches_not_active = torch.zeros(self.dict_size)
        self.block_indices = block_indices
        self.top_k = top_k
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        if self.W_enc.size(-1) == 0:
            self.W_dec = nn.Parameter(torch.empty(self.dict_size, self.act_size))
            self.W_enc = nn.Parameter(torch.empty(self.act_size, self.dict_size))

    def preprocess_input(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x = x - x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x = x / (x_std + 1e-5)
        return x, x_mean, x_std

    def postprocess_output(self, x_reconstruct, x_mean, x_std):
        x_reconstruct = x_reconstruct * x_std + x_mean
        return x_reconstruct

    def create_block_mask(self, tensor):
        batch, seq_len, n = tensor.shape
        if seq_len == 0:
            return torch.zeros((batch, 0, 0), dtype=torch.bool, device=tensor.device)
        diff_indicator = torch.zeros((batch, seq_len), dtype=torch.bool, device=tensor.device)
        diff_indicator[:, 0] = True
        if seq_len > 1:
            same_as_prev = (tensor[:, 1:] == tensor[:, :-1]).all(dim=-1)
            diff_indicator[:, 1:] = ~same_as_prev
        block_ids = diff_indicator.cumsum(dim=-1)
        mask = block_ids.unsqueeze(2) == block_ids.unsqueeze(1)
        return mask

    def forward(self, x, past_key_values=None):
        x, x_mean, x_std = self.preprocess_input(x)
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc)
        acts = self.scale * acts
        split_x = acts.split(self.block_indices, dim=-1)
        masks = []
        block_att_masks = []
        feature_split = []
        i = 0
        for sub_x, k in zip(split_x, self.top_k):
            _, indices = torch.topk(sub_x, min(k, sub_x.size(-1)), dim=-1)
            zero_mask = sub_x != 0
            mask = torch.zeros_like(sub_x)
            mask.scatter_(dim=-1, index=indices, value=1)
            mask = mask.bool() & zero_mask
            masks.append(mask)
            att_mask = self.create_block_mask(indices)
            if past_key_values is not None:
                if indices.size(-2) != 1:
                    indices = indices[:, -2, :].unsqueeze(1).repeat(1, att_mask[0, -2].tolist().count(True), 1)
                past_key_values.update_f(indices, i, self.top_k)
            block_att_masks.append(att_mask)
            feature_split.append(sub_x * mask)
            i += 1
        total_mask = torch.cat(masks, dim=-1)
        acts = acts * total_mask
        feature_split = [f @ self.W_dec[start: start + size] for start, size, f in
                         zip([0] + list(np.cumsum(self.block_indices)[:-1]), self.block_indices, feature_split)]
        features = acts @ self.W_dec + self.b_dec
        features = self.postprocess_output(features, x_mean, x_std)
        l0_norm = [(acts_topk > 0).float().sum(-1).mean() for acts_topk in acts.split(self.block_indices, dim=-1)]
        output = {
            'features': features,
            'l0_norm': l0_norm,
            'features_split': feature_split,
            'block_att_masks': block_att_masks
        }
        return output


# ============================================================
# Qwen2 base modules (from transformers 4.44.2)
# ============================================================

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# ============================================================
# Qwen2Attention with COREGEN DependencyEncoding
# ============================================================

class Qwen2Attention(nn.Module):
    """Qwen2 attention with COREGEN DependencyEncoding for block mask attention."""

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # COREGEN: masking layers use DependencyEncoding (second half of layers)
        num_layers = config.num_hidden_layers  # 28 for Qwen2.5-7B
        self.masking_layer = range(num_layers // 2, num_layers)
        self.activation_groups = [128, 1024, 16384]
        self.top_k_act = [1, 2, 8]
        self.register_buffer("m", get_alibi_slope(self.num_heads))

        # _use_block_mask is set by Qwen2Model based on model.use_block_mask
        self._use_block_mask = True

        if self.layer_idx == self.masking_layer[0]:
            self.dependency_encoding = DependencyEncoding(
                self.hidden_size, sum(self.activation_groups),
                self.activation_groups, self.top_k_act
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            line_positions=None,
            block_mask=None,
            position_embeddings=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        sparsity = None

        # --- COREGEN DependencyEncoding logic (at masking_layer[0]) ---
        if self.layer_idx == self.masking_layer[0]:
            if past_key_value is None or (q_len != 1) or \
                    (q_len == 1 and past_key_value.tokens[0].tolist()[-1] == 13):
                pos_embeddings = torch.zeros_like(hidden_states)
                if past_key_value is not None and q_len == 1 and past_key_value.tokens[0].tolist()[-1] == 13:
                    line_positions = torch.tensor([[0]], device='cuda')
                safe_indices = line_positions.clone()
                safe_indices[safe_indices == q_len] = 0
                expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
                gathered = torch.gather(hidden_states, 1, expanded_indices)
                outputs = self.dependency_encoding(gathered, past_key_value)
                encodings = outputs['features']

                repeat_indices = []
                mask = line_positions != -1
                for i in range(bsz):
                    valid_indices = line_positions[i][mask[i]]
                    if valid_indices.size(0) != 0:
                        repeat_idx = valid_indices[1:] - valid_indices[:-1]
                        repeat_idx = torch.cat(
                            [repeat_idx,
                             torch.tensor([hidden_states.size(-2) - valid_indices[-1]], device='cuda')])
                        repeat_indices.append(repeat_idx)
                        pos_embeddings[i, valid_indices[0]:] = pos_embeddings[i, valid_indices[0]:] + encodings[i,
                                                                                                      :len(valid_indices)
                                                                                                      ].repeat_interleave(
                            repeat_idx, dim=0)
                    position_embeddings = pos_embeddings
                if use_cache:
                    if q_len != 1:
                        past_key_value.lde = pos_embeddings[:, -2].unsqueeze(-2)
                    else:
                        past_key_value.lde = pos_embeddings[:, -1].unsqueeze(-2)

                # Block mask construction (only when use_block_mask=True, i.e. Stage 2)
                if self._use_block_mask:
                    feature_splits = outputs['features_split']
                    block_att_masks = outputs['block_att_masks']
                    block_masks = []
                    for dim, (feature_split, block_att_mask) in enumerate(zip(feature_splits, block_att_masks)):
                        feature_split_seq = torch.zeros_like(hidden_states)
                        b_mask = torch.zeros(bsz, self.num_heads, q_len, q_len, device=hidden_states.device, dtype=bool)
                        for i in range(bsz):
                            valid_indices = line_positions[i][mask[i]]
                            feature_split_seq[i, valid_indices[0]:] = feature_split_seq[i,
                                                                      valid_indices[0]:] + feature_split[i,
                                                                                           :len(valid_indices)].repeat_interleave(
                                repeat_indices[i], dim=0)
                            b_mask[i, :, valid_indices[0]:, valid_indices[0]:] = b_mask[i, :, valid_indices[0]:,
                                                                                 valid_indices[0]:] + block_att_mask[i,
                                                                                                      :len(valid_indices)].repeat_interleave(
                                repeat_indices[i], dim=0).repeat_interleave(repeat_indices[i], dim=1)
                        feature_q = self.q_proj(feature_split_seq)
                        feature_k = self.k_proj(feature_split_seq)
                        feature_q = feature_q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                        feature_k = feature_k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                        feature_k = repeat_kv(feature_k, self.num_key_value_groups)
                        if past_key_value is not None:
                            feature_q, feature_k = past_key_value.update_de(feature_q, feature_k, dim,
                                                                            self.activation_groups)
                        d_score = torch.matmul(feature_q, feature_k.transpose(2, 3))

                        causal_mask_de = torch.triu(torch.ones(q_len, q_len, dtype=torch.bool, device=d_score.device),
                                                    diagonal=1)
                        d_score = d_score + attention_mask[1][:, :, :, :q_len]
                        if past_key_value is not None:
                            final_mask = dynamic_topk_attention(d_score, training_length=1024, max_k=900,
                                                               is_prefilling=q_len != 1)
                        else:
                            final_mask = dynamic_topk_attention(d_score, training_length=1024, max_k=900,
                                                               is_prefilling=q_len != 1)
                        if past_key_value is not None and q_len == 1 and past_key_value.tokens[0].tolist()[-1] == 13:
                            b_mask = torch.zeros(bsz, 1, 1, past_key_value.tokens.size(-1), dtype=bool, device='cuda')
                            b_mask[:, :, :,
                            -past_key_value.find_position_difference(past_key_value.f[dim].size(-2)):] = True
                        final_mask = torch.logical_and(final_mask, ~b_mask)
                        final_mask = torch.logical_or(final_mask, causal_mask_de)
                        block_masks.append(final_mask)
                    block_mask = torch.any(torch.stack(block_masks), dim=0)
                    block_mask = block_mask.float()
                    block_mask[block_mask.bool()] = torch.finfo(hidden_states.dtype).min
                    if q_len == 1 and use_cache:
                        lbm = past_key_value.lbm
                        new_mask = torch.zeros(bsz, self.num_heads, lbm.size(-1) + 1, lbm.size(-1) + 1,
                                               dtype=block_mask.dtype, device=block_mask.device)
                        new_mask[:, :, lbm.size(-1), :lbm.size(-1)] = lbm[:, :, lbm.size(-1) - 1, :lbm.size(-1)]
                        new_mask[:, :, :lbm.size(-1), lbm.size(-1)] = lbm[:, :, :lbm.size(-1), lbm.size(-1) - 1]
                        new_mask[:, :, -1] = block_mask[:, :, 0, :]
                        new_mask += attention_mask[1]
                        block_mask = new_mask
                        past_key_value.lbm = block_mask
                    if use_cache:
                        past_key_value.lbm = block_mask
            elif q_len == 1 and past_key_value.tokens[0].tolist()[-1] != 13:
                position_embeddings = past_key_value.lde
                if self._use_block_mask:
                    for dim in range(len(self.activation_groups)):
                        _, feature_k = past_key_value.update_de(None, past_key_value.de_k[dim][:, :, -1,].unsqueeze(-2),
                                                                dim, self.activation_groups)
                    block_mask = past_key_value.lbm
                    new_mask = torch.zeros(bsz, self.num_heads, block_mask.size(-1) + 1, block_mask.size(-1) + 1,
                                           dtype=block_mask.dtype, device=block_mask.device)
                    new_mask[:, :, block_mask.size(-1), :block_mask.size(-1)] = block_mask[:, :, block_mask.size(-1) - 1,
                                                                                :block_mask.size(-1)]
                    new_mask[:, :, :block_mask.size(-1), block_mask.size(-1)] = block_mask[:, :, :block_mask.size(-1),
                                                                                block_mask.size(-1) - 1]
                    new_mask += attention_mask[1]
                    past_key_value.lbm = new_mask
                    block_mask = new_mask

        # Add dependency encoding to hidden states for masking layers
        if self.layer_idx in self.masking_layer and position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        # --- Standard attention computation with dual RoPE ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, 'get_usable_length'):
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            elif hasattr(past_key_value, 'get_seq_length'):
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # position_ids is [standard_pos_ids, relative_pos_ids] from Qwen2Model
        standard_position_ids = position_ids[0]
        relative_position_ids = position_ids[1] if len(position_ids) > 1 else position_ids[0]

        if past_key_value is not None and use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        if q_len == 1:
            # Inference: relative position RoPE
            current_pos = standard_position_ids[0, 0].item()
            inf_kv_seq_len = key_states.shape[2]
            full_position_ids = torch.arange(
                current_pos - inf_kv_seq_len + 1, current_pos + 1,
                device=standard_position_ids.device, dtype=standard_position_ids.dtype
            ).unsqueeze(0).expand(standard_position_ids.shape[0], -1)
            rel_pos = (standard_position_ids[:, -1:] - full_position_ids).clip(max=512)
            cos_rel, sin_rel = self.rotary_emb(value_states, seq_len=max(513, kv_seq_len))
            cos_rel = cos_rel[rel_pos].squeeze(1)  # (batch, kv_seq, head_dim)
            sin_rel = sin_rel[rel_pos].squeeze(1)
            cos_rel = cos_rel.unsqueeze(1)  # (batch, 1, kv_seq, head_dim)
            sin_rel = sin_rel.unsqueeze(1)
            key_states_rope = (key_states * cos_rel) + (rotate_half(key_states) * (-sin_rel))

            key_states_rope = repeat_kv(key_states_rope, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)
            attn_weights = torch.matmul(query_states, key_states_rope.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            # Training / prefilling: dual RoPE
            query_states1, key_states1 = apply_rotary_pos_emb(query_states, key_states, cos, sin,
                                                               standard_position_ids)
            window_position_ids = torch.full_like(standard_position_ids, 512)
            cos_win, sin_win = self.rotary_emb(value_states, seq_len=max(513, kv_seq_len))
            query_states2, _ = apply_rotary_pos_emb(query_states, key_states, cos_win, sin_win, window_position_ids)

            key_states1 = repeat_kv(key_states1, self.num_key_value_groups)
            key_states2 = repeat_kv(key_states, self.num_key_value_groups)
            value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights1 = torch.matmul(query_states1, key_states1.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights2 = torch.matmul(query_states2, key_states2.transpose(2, 3)) / math.sqrt(self.head_dim)
            rectified_mask = (standard_position_ids[:, -q_len:, None] - standard_position_ids[:, None]).abs() < 512
            attn_weights = torch.where(rectified_mask, attn_weights1, attn_weights2)

        # Apply attention mask
        if block_mask is not None:
            attn_mask = attention_mask[1][:, :, :, :q_len] + block_mask
        else:
            attn_mask = attention_mask[1]

        if attn_mask is not None:
            causal_mask = attn_mask[:, :, :, :key_states.shape[-2] if q_len > 1 else value_states_rep.shape[-2]]
            if use_cache and q_len == 1:
                causal_mask = causal_mask[:, :, -1].unsqueeze(-2)
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states_rep)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, sparsity, position_embeddings, block_mask


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
}


# ============================================================
# Qwen2DecoderLayer with COREGEN extra outputs
# ============================================================

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            line_positions=None,
            block_mask=None,
            position_embeddings=None,
            **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value, sparsity, position_embeddings, block_mask = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            line_positions=line_positions,
            block_mask=block_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        outputs += (sparsity,)
        outputs += (position_embeddings,)
        outputs += (block_mask,)
        return outputs


# ============================================================
# Qwen2PreTrainedModel
# ============================================================

class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ============================================================
# Qwen2Model with COREGEN mask construction
# ============================================================

class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Stage 1: use_block_mask=False (train DE only, no mask)
        # Stage 2: use_block_mask=True  (LoRA finetuning with mask)
        self.use_block_mask = True
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def window_attention_mask(self, seq_len, window_size):
        row_idx = torch.arange(seq_len, device=self.device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=self.device).unsqueeze(0)
        sink_mask = col_idx < 1
        window_start = torch.clamp(row_idx - window_size + 1, min=4)
        window_mask = (col_idx >= window_start) & (col_idx <= row_idx)
        mask = sink_mask | window_mask
        return mask

    def construct_mask_matrix(self, anchor_mask):
        total_size = anchor_mask.size(-1)
        batch_size = anchor_mask.size(0)
        rows, cols = anchor_mask.nonzero(as_tuple=True)
        unique_rows, counts = torch.unique_consecutive(rows, return_counts=True)
        grouped_indices = torch.split(cols, counts.tolist())
        indices = []
        if unique_rows.size(-1) != batch_size:
            unique_count = 0
            for i in range(batch_size):
                if i not in unique_rows:
                    indices.append(torch.tensor([]))
                else:
                    indices.append(grouped_indices[unique_count])
                    unique_count += 1
        else:
            indices = [id for id in grouped_indices]

        matrix_size = []
        for index in indices:
            if index.tolist():
                index_all = torch.cat(
                    [torch.tensor([-1], device=index.device), index,
                     torch.tensor([total_size - 1], device=index.device)], dim=-1)
                diff = index_all[1:] - index_all[:-1]
            else:
                diff = torch.tensor([])
            matrix_size.append(diff)

        batched_sizes = [group.tolist() for group in matrix_size]
        result_matrix = torch.zeros(batch_size, total_size, total_size, device='cuda')
        anchor_matrix = torch.zeros(batch_size, total_size, total_size, device='cuda')
        for idx, sizes in enumerate(batched_sizes):
            start_index = 0
            for size in sizes:
                tri_matrix = torch.tril(torch.ones(size, size))
                result_matrix[idx, start_index:start_index + size, start_index:start_index + size] = tri_matrix
                anchor_matrix[idx, start_index:start_index + size, start_index:start_index + size] = tri_matrix
                start_index += size
                if start_index != total_size:
                    anchor_matrix[idx, :, start_index - 1] = 1.
        return torch.tril(result_matrix), torch.tril(anchor_matrix)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, CodeDynamicCache) and not self.training:
            return_legacy_cache = True
            past_key_values = CodeDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        if not use_cache:
            past_key_values = None

        # Track tokens for line-boundary detection
        if past_key_values is not None:
            if past_key_values.tokens is None:
                past_key_values.tokens = input_ids
            else:
                past_key_values.tokens = torch.cat([past_key_values.tokens, input_ids], dim=-1)
                input_ids = past_key_values.tokens

        # COREGEN: construct anchor mask and line positions
        # Token 13 = newline (\n) in most tokenizers; also use eos_token_id
        anchor_mask = torch.bitwise_or(input_ids == 13, input_ids == 1)
        anchor_mask = anchor_mask.int()
        deep_mask, anchor_mask_matrix = self.construct_mask_matrix(anchor_mask)
        deep_mask = deep_mask.unsqueeze(1)
        anchor_mask_matrix = anchor_mask_matrix.unsqueeze(1)
        deep_mask = (~deep_mask.bool()).float()
        deep_mask[deep_mask.bool()] = torch.finfo(inputs_embeds.dtype).min
        anchor_mask_matrix = (~anchor_mask_matrix.bool()).float()
        anchor_mask_matrix[anchor_mask_matrix.bool()] = torch.finfo(inputs_embeds.dtype).min
        window_mask = self.window_attention_mask(input_ids.size(-1), 512).bool()
        ca_mask = torch.tril(
            torch.ones(input_ids.size(0), 1, input_ids.size(1), input_ids.size(1), device=input_ids.device))
        window_mask = torch.bitwise_or(
            ~window_mask.view(1, 1, input_ids.size(1), input_ids.size(1)),
            ~ca_mask.bool()).float()
        window_mask[window_mask.bool()] = torch.finfo(inputs_embeds.dtype).min
        causal_mask = [deep_mask, causal_mask, window_mask]

        # Compute line positions for DependencyEncoding
        row_indices, col_indices = torch.where(torch.bitwise_or(input_ids == 13, input_ids == 1))
        max_indices_per_row = 200
        if not self.training:
            max_indices_per_row = min(int(row_indices.size(0) / max(input_ids.size(0), 1)), 800)
        line_positions = torch.full((input_ids.size(0), max_indices_per_row + 1), input_ids.size(-1), dtype=torch.long,
                                    device=input_ids.device)
        counts = torch.zeros(input_ids.size(0), dtype=torch.long)
        for row, col in zip(row_indices, col_indices):
            count = counts[row]
            if count < max_indices_per_row:
                line_positions[row, count] = col
                counts[row] += 1

        # Compute relative position IDs for dual RoPE
        line_positions_ = torch.cat(
            [torch.tensor([0], device=hidden_states.device).unsqueeze(0).repeat(input_ids.size(0), 1), line_positions],
            dim=-1)
        indices = torch.cat([line_positions_[:, 1:] - line_positions_[:, :-1]], dim=-1)
        position_ids_ = torch.stack(
            [torch.cat([torch.arange(0, i) for i in ids if i > 0], dim=-1) for ids in indices],
            dim=0).clip(0, 127)
        position_ids = [position_ids, position_ids_]

        position_embeddings = None

        # Propagate use_block_mask flag to attention layers
        for layer in self.layers:
            layer.self_attn._use_block_mask = self.use_block_mask

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        sparsity = []
        block_mask = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    line_positions,
                    block_mask,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    line_positions=line_positions,
                    block_mask=block_mask,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            sparsity.append(layer_outputs[-3])
            position_embeddings = layer_outputs[-2]
            block_mask = layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


# ============================================================
# Qwen2ForCausalLM
# ============================================================

class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
            cache_position=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = past_key_values.get_max_length()
            else:
                past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            if max_cache_length is not None and attention_mask is not None and \
                    max_cache_length + input_ids.shape[1] > attention_mask.shape[1]:
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update({
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
