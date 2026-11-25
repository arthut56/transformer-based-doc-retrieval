import os
import torch


class VectorIndex:
    def __init__(self, vectors: torch.Tensor, ids: list[str]):
        assert vectors.size(0) == len(ids), 'Need exactly one id per vector'
        self.vectors = vectors
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def scores(self, queries: torch.Tensor) -> torch.Tensor:
        """Similarity of every query (rows) to every stored passage (columns)."""
        return queries @ self.vectors.T

    def search(self, queries: torch.Tensor, k: int):
        top = torch.topk(self.scores(queries), min(k, len(self)), dim=1)
        return top.values, top.indices

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'vectors': self.vectors.cpu(), 'ids': self.ids}, path)

    @classmethod
    def load(cls, path: str, device):
        stored = torch.load(path, map_location=device)
        return cls(stored['vectors'], stored['ids'])
