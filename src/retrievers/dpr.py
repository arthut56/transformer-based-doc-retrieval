from transformers import AutoTokenizer, AutoModel
import torch
from src.retrievers.models.encoder import TransformerEncoder

class DPR:
    def __init__(self, cfg, documents = None, index_path = None):
        super().__init__(documents, index_path)
        self.cfg = cfg
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
        self.query_encoder, self.document_encoder = AutoModel.from_pretrained(cfg.model.encoder)
        self.document_embedding = self.build_document_embeddings()
        self.__load_models()

    def tokenize_query(self, texts, max_length):
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

        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(query_weights_path)

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            document_encoder.load_state_dict(document_weights_path)

