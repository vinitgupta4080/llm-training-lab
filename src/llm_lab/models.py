import math

import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.table(tokens)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, context: int, dropout: float = 0.0):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context, context, dtype=torch.bool)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        reshape = lambda z: z.view(batch, time, self.heads, self.head_width).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.mask[:time, :time], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, time, width)
        return self.out(attended)


class Block(nn.Module):
    def __init__(self, width: int, heads: int, context: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, context, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=259, context=64, width=128, heads=4, layers=4, dropout=0.0):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads, context, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, time = tokens.shape
        if time > self.context:
            raise ValueError(f"sequence length {time} exceeds context {self.context}")
        positions = torch.arange(time, device=tokens.device)
        logits = self.lm_head(self.norm(self.blocks(self.token(tokens) + self.position(positions))))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, count: int, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(count):
            logits, _ = self(tokens[:, -self.context :])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            tokens = torch.cat((tokens, torch.multinomial(probs, 1)), dim=1)
        return tokens

