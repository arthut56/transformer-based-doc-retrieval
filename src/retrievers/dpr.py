from transformers import AutoTokenizer
import torch
from src.retrievers.models.encoder import TransformerEncoder
from src.retrievers.base_retriever import Retriever
from src.retrievers.utils import get_device

class DPR(Retriever):
    def __init__(self, cfg, documents=None, index_path=None):
        super().__init__(documents, index_path)  # 1. gives self.index
        self.cfg = cfg
        self.device = get_device() if cfg.device == 'auto' else torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        self.query_encoder, self.document_encoder = self.__load_models()
        self.document_embedding = self.build_document_embeddings()

    def tokenize(self, texts, max_length):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt")['input_ids'] #bert uses pad_id 0 (mask ignores pad when attentioning or applying attention? Grammar?)
    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        m = self.cfg.model
        def build():
            return TransformerEncoder(vocab_size=self.tokenizer.vocab_size,
                                      pad_id=self.tokenizer.pad_token_id,
                                      pooling_type=m.pooling_type,
                                      dropout=m.dropout,
                                      n_layers=m.n_layers,n_heads=m.n_heads,
                                      emb_dim=m.emb_dim,
                                      hidden_dim=m.hidden_dim,
                                      feed_forward_dim=m.feed_forward_dim)
        query_encoder = build()
        document_encoder = build()

        query_weights_path = m.get('query_encoder_weights_path', None)
        if query_weights_path:
            query_encoder.load_state_dict(torch.load(query_weights_path, map_location=self.device))

        document_weights_path = m.get('document_encoder_weights_path', None)
        if document_weights_path:
            document_encoder.load_state_dict(torch.load(document_weights_path, map_location=self.device))

        return query_encoder, document_encoder

    def encode(self, texts, encoder, max_length, batch_size = 64):
        out = []
        for i in range (0, len(texts), batch_size):
            ids = self.tokenize(texts[i:i+batch_size], max_length).to(self.device)
            out.append(encoder(ids))
            return torch.cat(out)

    def build_document_embeddings(self):
        texts = [doc.GetText() for doc in self.index.GetDocuments()]
        return self.encode(texts, self.document_encoder, self.cfg.model.doc_max_length)

    def CalculateScores(self, queries: list[str]):
        q = self.encode(queries, self.query_encoder, self.cfg.model.query_max_length)
        return (q@self.document_embedding.T).cpu()
