import os
import pickle
from abc import ABC, abstractmethod
from src.data_utils.dataset import DocumentDataset
import torch
class Retriever(ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        if not index_path is None:
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__BuildIndex(documents)
    
    def __BuildIndex(self, documents: list[dict]):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        index = DocumentDataset(documents)
        return index
    
    def __LoadIndex(self, index_path: str):
        """
        @param index_path: The path to the pre-computed index.
        """
        file = open(index_path, 'rb')
        index = pickle.load(file)
        file.close()
        return index
    
    def SaveIndex(self, index_path: str):
        """
        @param index_path: The path to save the pre-computed index.
        """
        if not os.path.exists(os.path.dirname(index_path)):
            os.makedirs(os.path.dirname(index_path))
        file = open(index_path, 'wb')
        pickle.dump(self.index, file)
        file.close()

        
    @abstractmethod
    def CalculateScores(self, query: str):
        raise NotImplementedError("Must overwrite")
    
    def Lookup(self, queries: list[str], k: int, batch_size: int = 1024) -> list:
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        """
        docs = self.index.GetDocuments()
        results = []
        for i in range(0, len(queries), batch_size):
            scores = torch.as_tensor(self.CalculateScores(queries[i:i + batch_size]))
            top_idx = torch.topk(scores, min(k, scores.size(1)), dim=1).indices
            results.extend([docs[j] for j in row] for row in top_idx.tolist())
        return results