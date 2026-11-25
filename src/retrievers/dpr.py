import os
from transformers import AutoTokenizer
import torch
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.models.pretrained_encoder import PretrainedEncoder
from src.retrievers.base_retriever import Retriever
from src.retrievers.utils import get_device
from src.retrievers.vector_index import VectorIndex

class DPR(Retriever):
    def __init__(self, cfg, documents=None, index_path=None):
        super().__init__(documents, index_path)  # 1. gives self.index
        self.cfg = cfg
        self.device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        self.query_encoder, self.document_encoder = self.__load_models()
        self.vector_index = None
        # The vector index is stored next to the document encoder weights it was built from
        document_weights_path = cfg.model.get('document_encoder_weights_path', None)
        self.index_path = os.path.join(os.path.dirname(document_weights_path), 'doc_index.pt') if document_weights_path else None

    def tokenize(self, texts, max_length):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")['input_ids'] #bert uses pad_id 0 (mask ignores pad when attentioning or applying attention? Grammar?)
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        m = self.cfg.model
        def build():
            if m.get('type', 'scratch') == 'pretrained':
                return PretrainedEncoder(model_name=m.pretrained_name,
                                         pad_id=self.tokenizer.pad_token_id,
                                         pooling_type=m.pooling_type,
                                         hidden_dim=m.get('hidden_dim', None),
                                         normalize=m.get('normalize', False))
            return TransformerEncoder(vocab_size=self.tokenizer.vocab_size,
                                      pad_id=self.tokenizer.pad_token_id,
                                      pooling_type=m.pooling_type,
                                      dropout=m.dropout,
                                      n_layers=m.n_layers,n_heads=m.n_heads,
                                      emb_dim=m.emb_dim,
                                      hidden_dim=m.hidden_dim,
                                      feed_forward_dim=m.feed_forward_dim,
                                      normalize=m.get('normalize', False),
                                      norm_first=m.get('norm_first', False))
        query_encoder = build()
        document_encoder = build()

        query_weights_path = m.get('query_encoder_weights_path', None)
        if query_weights_path:
            query_encoder.load_state_dict(torch.load(query_weights_path, map_location=self.device))

        document_weights_path = m.get('document_encoder_weights_path', None)
        if document_weights_path:
            document_encoder.load_state_dict(torch.load(document_weights_path, map_location=self.device))

        return query_encoder.to(self.device).eval(), document_encoder.to(self.device).eval() ## can't just do return query_encoder, document encoder
    @torch.no_grad()
    def encode(self, texts, encoder, max_length, batch_size = 64):
        out = []
        for i in range (0, len(texts), batch_size):
            ids = self.tokenize(texts[i:i+batch_size], max_length).to(self.device)
            out.append(encoder(ids))
        return torch.cat(out)

    def _load_stored_index(self, ids):
        """Returns the stored index, or None if there is none or it is out of date."""
        if not (self.index_path and os.path.exists(self.index_path)):
            return None
        # Rebuild if the encoder weights changed after the index was written
        weights = os.path.join(os.path.dirname(self.index_path), 'document_encoder.pt')
        if os.path.exists(weights) and os.path.getmtime(weights) > os.path.getmtime(self.index_path):
            return None
        stored = VectorIndex.load(self.index_path, self.device)
        return stored if stored.ids == ids else None  # a different corpus was indexed

    def build_vector_index(self):
        docs = self.index.GetDocuments()
        ids = [doc.GetId() for doc in docs]
        stored = self._load_stored_index(ids)
        if stored is not None:
            print(f'Loaded vector index from {self.index_path}')
            return stored
        vectors = self.encode([doc.GetText() for doc in docs], self.document_encoder, self.cfg.model.doc_max_length)
        vector_index = VectorIndex(vectors, ids)
        if self.index_path:
            vector_index.save(self.index_path)
            print(f'Saved vector index ({len(vector_index)} passages) to {self.index_path}')
        return vector_index

    def Lookup(self, queries: list[str], k: int, batch_size: int = 1024) -> list:
        if self.vector_index is None:
            self.vector_index = self.build_vector_index()
        docs = self.index.GetDocuments()
        results = []
        for i in range(0, len(queries), batch_size):
            q = self.encode(queries[i:i + batch_size], self.query_encoder, self.cfg.model.query_max_length)
            _, top_idx = self.vector_index.search(q, k)
            results.extend([docs[j] for j in row] for row in top_idx.tolist())
        return results

    def CalculateScores(self, queries: list[str]):
        if self.vector_index is None:
            self.vector_index = self.build_vector_index()
        q = self.encode(queries, self.query_encoder, self.cfg.model.query_max_length)
        return self.vector_index.scores(q).cpu()
