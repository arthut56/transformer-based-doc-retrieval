import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PretrainedEncoder(nn.Module):
    def __init__(self, model_name: str, pad_id: int, pooling_type: str,
                 hidden_dim: int = None, normalize: bool = False):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, attn_implementation='eager')
        self.pad_id = pad_id
        self.pooling_type = pooling_type
        self.normalize = normalize
        size = self.model.config.hidden_size
        self.proj = nn.Linear(size, hidden_dim) if hidden_dim else nn.Identity()
        self._attentions = None

    def pool(self, x, mask):
        keep = mask.unsqueeze(-1)
        if self.pooling_type == 'cls':
            return x[:, 0]
        elif self.pooling_type == 'max':
            return x.masked_fill(keep == 0, -math.inf).max(dim=1).values
        elif self.pooling_type == 'mean':
            keep = keep.float()
            return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        raise ValueError(f'Unknown pooling_type: {self.pooling_type}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.pad_id).long()
        out = self.model(input_ids=x, attention_mask=mask, output_attentions=not self.training)
        self._attentions = out.attentions
        x = self.proj(self.pool(out.last_hidden_state, mask))
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x

    def attention_maps(self):
        return list(self._attentions)
