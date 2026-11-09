import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from src.retrievers.models import layers
import math

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, pooling_type: str, dropout: float,
                 n_layers: int, n_heads: int, emb_dim: int, hidden_dim: int, feed_forward_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.emb_dim =    emb_dim
        self.hidden_dim = hidden_dim
        self.feed_forward_dim = feed_forward_dim

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.pooling_type = pooling_type
        self.dropout = dropout

        self.setup_model()

    def setup_model(self):
        """
        Instantiate the model.
        """
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_id)
        self.position_embedding = layers.PositionalEncoding(self.emb_dim, self.dropout)
        self.layers = nn.ModuleList([layers.EncoderLayer(self.emb_dim, self.n_heads,
                                                         self.feed_forward_dim, self.dropout) for _ in range(self.n_layers)])
        self.proj = nn.Linear(self.emb_dim, self.hidden_dim) #TODO: Compress dimensions, We could omit this in future testing to see if it improves performance or not

    def pool(self, x, mask):
        """
        Given a batch_size x sequence_length x emb_dim tensor, pool the input along the sequence_length dimension 
        Returns: A pooled vector of shape batch_size x emb_dim
        """
        keep = mask.squeeze(1).unsqueeze(-1)
        if self.pooling_type == 'cls':
            return x[:, 0]

        elif self.pooling_type == 'max':
            masked = x.masked_fill(keep == 0, -math.inf)
            x = masked.max(dim=1).values
            return x
        elif self.pooling_type == 'mean':
            keep = keep.float()
            return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        raise ValueError(f'Unknown pooling_type: {self.pooling_type}')

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Run a forward pass on the model
        """
        mask = (x != self.pad_id).unsqueeze(1) # Bx1xL I think
        x = self.token_embedding(x) * math.sqrt(self.emb_dim)
        x = self.position_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.proj(self.pool(x, mask))
        return x

    def attention_maps(self):
        return [layer.self_attn.attn for layer in self.layers]