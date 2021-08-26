import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# constants

ABS_MAX_STEPS = 100

# helper functions

def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, h, device = x.shape[1], self.heads, x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = device).triu(j - i + 1).bool()
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# pondering classes and helper functions

def exclusive_cumprod(t):
    return F.pad(t.cumprod(dim = -1), (1, -1), value = 1.)

def calc_geometric_seq(l):
    return exclusive_cumprod(1 - l) * l

# main class

class Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal))
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult))

        self.to_halt_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, 1),
            Rearrange('b () -> b')
        )

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x

        return x, self.to_halt_logits(x)

class PonderTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len,
        causal = True,
        dim_head = 64,
        heads = 8,
        ponder_kl_div_loss_weight = 0.01,
        ponder_lambda_p = 0.2,
        ponder_max_steps = None,
        ponder_epsilon = 0.05
    ):
        super().__init__()
        self.seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # calculate max steps

        if exists(ponder_max_steps):
            self.train_max_steps = ponder_max_steps
        else:
            thres = 1 - ponder_epsilon
            halt_probs = calc_geometric_seq(torch.full((ABS_MAX_STEPS,), ponder_lambda_p))
            cum_halt_probs = halt_probs.cumsum(dim = 0)
            self.train_max_steps = (cum_halt_probs < thres).sum().item()

        # pondering block

        self.block = Block(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            causal = causal
        )

        # hidden state to 'y' - output

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        if self.training:
            # training mode

            for _ in range(self.train_max_steps):
                x, halt_logits = self.block(x)

            return self.to_logits(x)
        else:
            # evaluation mode

            for _ in range(self.train_max_steps):
                x, halt_logits = self.block(x)

            return self.logits(x)
