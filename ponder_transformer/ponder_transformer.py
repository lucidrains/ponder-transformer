import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
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

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = device).triu(j - i + 1).bool()
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# pondering classes and helper functions

def pad_to(t, padding, dim = -1, value = 0.):
    if dim > 0:
        dim = dim - t.ndim
    zeroes = -dim - 1
    return F.pad(t, (*((0, 0) * zeroes), *padding), value = value)

def safe_cumprod(t, eps = 1e-10, dim = -1):
    t = torch.clip(t, min = eps, max = 1.)
    return torch.exp(torch.cumsum(torch.log(t), dim = dim))

def exclusive_cumprod(t, dim = -1):
    cum_prod = safe_cumprod(t, dim = dim)
    return pad_to(cum_prod, (1, -1), value = 1., dim = dim)

def calc_geometric(l, dim = -1):
    return exclusive_cumprod(1 - l, dim = dim) * l

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
        self.causal = causal
        self.attn = PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal))
        self.ff = PreNorm(dim, FeedForward(dim = dim, mult = ff_mult))

        self.to_halt_logits = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... () -> ...')
        )

    def forward(self, x, mask = None):
        x = self.attn(x, mask = mask) + x
        x = self.ff(x) + x

        if self.causal:
            denom = torch.arange(x.shape[-2], device = x.device)
            denom = rearrange(denom, 'n -> () n ()')
            halt_input = x.cumsum(dim = 1) / (denom + 1)
        else:
            halt_input = x.mean(dim = 1)

        halt_logits = self.to_halt_logits(halt_input)

        return x, halt_logits

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
        ponder_epsilon = 0.05,
        eps = 1e-20
    ):
        super().__init__()
        self.eps = eps
        self.causal = causal
        self.seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # calculate max steps

        thres = 1 - ponder_epsilon
        halt_probs = calc_geometric(torch.full((ABS_MAX_STEPS,), ponder_lambda_p))
        cum_halt_probs = halt_probs.cumsum(dim = 0)
        self.train_max_steps = (cum_halt_probs < thres).sum().item()

        self.ponder_lambda_p = ponder_lambda_p
        self.ponder_kl_div_loss_weight = ponder_kl_div_loss_weight

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

    def forward(self, x, *, labels = None, mask = None):
        n, device, eps, max_steps, causal = x.shape[1], x.device, self.eps, self.train_max_steps, self.causal
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        if self.training:
            assert exists(labels), 'labels must be passed in during training'

            hiddens = []
            halting_logits = []

            # training mode

            for _ in range(max_steps):
                x, halt_logits = self.block(x)

                hiddens.append(x)
                halting_logits.append(halt_logits)

            # stack halting probs (lambda) and y

            halting_logits = torch.stack(halting_logits, dim = 1)
            halting_probs = calc_geometric(halting_logits.sigmoid(), dim = 1)

            hiddens = torch.stack(hiddens, dim = 1)
            logits = self.to_logits(hiddens)

            # calculate kl div with geometric prior

            geometric_dist = calc_geometric(torch.full((max_steps,), self.ponder_lambda_p, device = device))

            if self.causal:
                geometric_dist = repeat(geometric_dist, 'l -> (l n)', n = n)
                halting_probs = rearrange(halting_probs, '... l n -> ... (l n)')

            kl_div_loss = F.kl_div(
                torch.log(geometric_dist + eps),
                halting_probs,
                None, None,
                'batchmean'
            )

            # calculate cross entropy loss

            labels = repeat(labels, 'b n -> b (l n)', l = max_steps)
            logits = rearrange(logits, 'b l n d -> b d (l n)')
            ce_loss = F.cross_entropy(logits, labels, ignore_index = 0)

            weighted_ce_loss = ce_loss * halting_probs

            # sum loss

            loss = weighted_ce_loss.mean() + self.ponder_kl_div_loss_weight * kl_div_loss.mean()
            return loss
        else:
            # evaluation mode

            hiddens = []
            halting_logits = []
            layer_halt = []

            for i in range(self.train_max_steps):
                is_last = i == (self.train_max_steps - 1)

                x, halt_logits = self.block(x)
                hiddens.append(x)

                if self.causal:
                    halt_logits = halt_logits[..., -1]

                halting_logits.append(halt_logits)

                # calculating halting probs

                halting_probs = torch.stack(halting_logits, dim = 1).sigmoid()
                p = calc_geometric(halting_probs, dim = 1)[:, -1]
                should_halt = torch.rand_like(p) <= p

                # stack the halting signal across layers and determine whether to stop early

                layer_halt.append(should_halt)

                # do not exit early if it is the last one

                if is_last:
                    continue

                # break if halting has been sampled for all layers

                layer_was_halted = torch.any(torch.stack(layer_halt), dim = 0)

                if torch.all(layer_was_halted):
                    break

            # calculate max number of layers

            max_num_layers = len(layer_halt)

            # stack the hiddens and the boolean tensor indicating halting for each layer

            hiddens = torch.stack(hiddens, dim = 1)
            layer_halt = torch.stack(layer_halt, dim = 1)

            # calculate the index of the first halt signal, and make it the last layer if none of them halted

            halt_layer_indices = (layer_halt.cumsum(dim = 1) == 0).sum(dim = 1).clamp(max = max_num_layers - 1)

            # select out the correct hidden layers to logits

            halt_layer_indices_expanded = repeat(halt_layer_indices, 'b -> b () n d', n = hiddens.shape[-2], d = hiddens.shape[-1])
            hiddens = hiddens.gather(1, halt_layer_indices_expanded)
            hiddens = rearrange(hiddens, 'b () n d -> b n d')

            return self.to_logits(hiddens), halt_layer_indices
