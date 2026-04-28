import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional

ADALN_EMBED_DIM = 256

FREQUENCY_EMBEDDING_SIZE = 256
MAX_PERIOD = 10000

def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)

class exp_embedder(nn.Module):
    def __init__(self, d_model):
        super(exp_embedder, self).__init__()
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    def forward(self, expression_val):
        B,T = expression_val.size()
        position = expression_val.reshape(B,-1,1)
        pos_enc = torch.zeros((B, T, self.d_model), device=position.device)
        pos_enc[:, :, 0::2] = torch.sin(position * self.div_term.to(position.device))
        pos_enc[:, :, 1::2] = torch.cos(position * self.div_term.to(position.device))

        return pos_enc

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    _attention_backend = None

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Dispatch
        from util.attention import dispatch_attention

        hidden_states = dispatch_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, backend=self._attention_backend
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class sc2Flow(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        hidden_size=256,
        num_heads=4,
        in_context_len=32,
        num_genes=2001,

        dim=768,
        n_layers=5,
        n_refiner_layers=5,
        n_heads=6,
        n_kv_heads=6,
        norm_eps=1e-5,
        qk_norm=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len

        # time and class embed
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)

        # linear embed
        self.exp_embedder = exp_embedder(d_model=dim)
        #self.x_embedder = BottleneckPatchEmbed(hidden_size, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        embedding_dim =  dim//n_heads//2
        self.gene_embed = nn.Embedding(num_genes, embedding_dim)

        # transformer
        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.c_refiner = nn.ModuleList(
            [
                TransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )

        self.final_layer = FinalLayer(dim, 1)

        # linear predict
        #self.final_layer = FinalLayer(hidden_size, 4, self.out_channels)

        #self.initialize_weights()

        self.temp_c_linear = nn.Linear(768, dim)
        self.d_embdding = nn.Parameter(torch.randn(1, dim))
    def initialize_weights(self):
        pass

    def forward(self, x, t, labels):
        """
        x: gene expression values
        labels: gene id, condition, dosage ...
        """
        # class and time embeddings

        c = labels['c_rep']
        g_id = labels['g_id']
        d = labels['dose']

        t = self.t_embedder(t)
        c = self.temp_c_linear(c)
        d = d.unsqueeze(1) @ self.d_embdding

        x_attn_mask = (g_id != 0).bool().to(t.device)
        g_emb = self.gene_embed(g_id)

        max_len = g_id.shape[1]

        x = self.exp_embedder(x)

        adaln_input = t.type_as(x)
        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, g_emb, adaln_input)


        unified = torch.cat([x, c.unsqueeze(1), d.unsqueeze(1)], dim=1)

        g_attn_mask = torch.ones([x_attn_mask.shape[0],1],dtype=torch.bool).to(t.device)
        d_attn_mask = torch.ones([x_attn_mask.shape[0],1],dtype=torch.bool).to(t.device)
        unified_attn_mask = torch.cat([x_attn_mask, g_attn_mask,d_attn_mask], dim=1)

        c_freqs_cis = torch.zeros(unified.shape[0], 1, 64).to(t.device)
        d_freqs_cis = torch.zeros(unified.shape[0], 1, 64).to(t.device)
        unified_freqs_cis = torch.cat([g_emb, c_freqs_cis,d_freqs_cis], dim=1)

        adaln_input = t.type_as(unified)

        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        output = self.final_layer(unified, adaln_input).squeeze(-1)

        return output[:,:max_len]