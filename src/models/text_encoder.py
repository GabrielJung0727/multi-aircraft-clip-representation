"""
Lightweight, self-contained text encoder (no external pretrained models).

Uses a hashing tokenizer + embedding + transformer-style projection to
produce fixed-length text embeddings for contrastive training.
"""

from __future__ import annotations

import re
from typing import Iterable, List

import torch
from torch import nn


def simple_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower()) or ["empty"]


def hash_tokens(tokens: Iterable[str], vocab_size: int) -> List[int]:
    return [hash(tok) % vocab_size for tok in tokens]


class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 8192, embed_dim: int = 512, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        hidden = hidden_dim or embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # token_ids: (B, T)
        embeds = self.embedding(token_ids)  # (B, T, D)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            embeds = embeds * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = embeds.sum(dim=1) / denom
        else:
            pooled = embeds.mean(dim=1)
        encoded = self.encoder(pooled)
        return self.proj(encoded)


class HashingTextEncoder:
    """Callable wrapper to encode raw text strings to embeddings without external models."""

    def __init__(self, vocab_size: int, embed_dim: int, device: torch.device) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.model = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim).to(device)
        self.model.eval()
        self.cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        uncached = [t for t in texts if t not in self.cache]
        if uncached:
            token_lists = [hash_tokens(simple_tokenize(t), self.vocab_size) for t in uncached]
            max_len = max(len(toks) for toks in token_lists)
            padded = []
            attn = []
            for toks in token_lists:
                pad_len = max_len - len(toks)
                padded.append(toks + [0] * pad_len)
                attn.append([1.0] * len(toks) + [0.0] * pad_len)
            token_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            attn_tensor = torch.tensor(attn, dtype=torch.float32, device=self.device)
            embeddings = self.model(token_tensor, attention_mask=attn_tensor).cpu()
            for text, emb in zip(uncached, embeddings):
                self.cache[text] = emb

        stacked = torch.stack([self.cache[t] for t in texts]).to(self.device)
        return stacked


__all__ = ["SimpleTextEncoder", "HashingTextEncoder", "simple_tokenize", "hash_tokens"]
