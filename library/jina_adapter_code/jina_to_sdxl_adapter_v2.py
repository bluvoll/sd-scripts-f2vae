import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# Adapted from NeuroSenko/ComfyUI_LLM_SDXL_Adapter
logger = logging.getLogger("JinaCLIP-SDXL-Adapter")

def pad_to_length(tensor, target_length, dim=1, value=0):
    """Universal function for padding tensors"""
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length

    padding = torch.full(
        pad_size,
        value,
        device=tensor.device,
        dtype=tensor.dtype
    )

    return torch.cat([tensor, padding], dim=dim)

class AttentionPooler(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True) # should use ExplicitMHA but...
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # x: (Batch, Seq_Len, Dim)
        batch_size = x.shape[0]
        q = self.query.expand(batch_size, -1, -1) # (B, 1, Dim)
        
        # nn.MultiheadAttention expects True for padding tokens, opposite of pytorch's SDPA
        # key_padding_mask should be True for tokens to ignore

        key_padding_mask = ~mask.bool() if mask is not None else None

        attn_out, _ = self.attn(
            q, x, x, 
            key_padding_mask=key_padding_mask
        )
        
        return self.norm(attn_out.squeeze(1)) # (B, Dim)

class ExplicitMultiheadAttention(nn.Module):
    """
    An explicit implementation of Multi-head Attention to ensure LoRA compatibility.
    Replaces the monolithic nn.MultiheadAttention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Separate unfused linear layers for Q, K, V and Output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, average_attn_weights=True):
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if need_weights:
            raise ValueError("need_weights=True is not supported by F.scaled_dot_product_attention.")

        # Prepare attn mask
        attn_mask = None
        if key_padding_mask is not None:
            # nn.MultiheadAttention expects True for padding tokens, opposite of pytorch's SDPA
            # key_padding_mask should be True for tokens to ignore

            # invert mask, pytorch's SDPA exoects True for valid tokens and False for padding
            # (batch_size, seq_len) to (batch_size, 1, 1, seq_len)
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout.p if self.training else 0.0

        # SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=dropout_p,
            is_causal=False
        )

        # reshape, project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, None

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ExplicitMultiheadAttention(
            dim, num_heads, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask=None):
        # Self-attention
        normed = self.norm1(x)

        if mask is not None:
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x

class JinaToSDXLAdapterV2(nn.Module):
    # Jina-clip-v2 to SDXL
    def __init__(self,
                 llm_dim=1024,           
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 n_attention_blocks=4,
                 num_heads=16,
                 dropout=0,
                 max_seq_len=539,
                 attn_pooling=True):
        super().__init__()
        self.attn_pooling = attn_pooling

        self.seq_projection = nn.Sequential(
            nn.Linear(llm_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim)
        )

        self.positional_embedding = nn.Embedding(max_seq_len, sdxl_seq_dim)

        self.attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_attention_blocks)
        ])

        if self.attn_pooling:
            self.attention_pooler = AttentionPooler(sdxl_seq_dim)
            self.pooled_projection = nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        else:
            # Projection from Jina-clip's 1024 to 1280
            self.pooled_projection = nn.Sequential(
                nn.Linear(llm_dim, llm_dim),
                nn.LayerNorm(llm_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(llm_dim, sdxl_pooled_dim)
            )

    def forward(self, jina_hidden_states, jina_mean_pooled_state, attention_mask=None):
        """
        jina_hidden_states: (Batch, Seq_Len, 1024)
        jina_mean_pooled_state: (Batch, 1024), used in V1 (attn_pooling=False), generated via Jina's mean_pooling
        attention_mask: (Batch, Seq_Len), 1 for valid tokens, 0 for padding
        """
        # Map 1024 -> 2048
        hidden_states = self.seq_projection(jina_hidden_states)

        seq_len = hidden_states.size(1)
        positions = torch.arange(seq_len, device=hidden_states.device)
        hidden_states = hidden_states + self.positional_embedding(positions).unsqueeze(0)

        # attention layers
        for block in self.attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        if self.attn_pooling:
            pooled_features = self.attention_pooler(hidden_states, attention_mask)
            pooled_output = self.pooled_projection(pooled_features)
        else:
            pooled_output = self.pooled_projection(jina_mean_pooled_state)

        return hidden_states, pooled_output