import hydra

from src.retrievers.bm25 import BM25
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.data_utils.query import make_queries
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR
from torch.utils.data import DataLoader
from src.training.trainer import Trainer, PairDataset

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries (SQuAD has no test split, so splits may be missing)
    splits = {'Train': corpus.train, 'Val': corpus.val, 'Test': corpus.test}
    queries = {name: make_queries(corpus, split) for name, split in splits.items() if split is not None}

    if cfg.retriever.method == 'bm25':
        retriever = BM25(documents=corpus.all_documents)
    elif cfg.retriever.method == 'dpr':
        retriever = DPR(cfg, documents=corpus.all_documents)
        if cfg.get('do_train', False):
            trainer = Trainer(retriever, cfg)
            loader = DataLoader(PairDataset(corpus.train), batch_size=cfg.train.batch_size,
                                shuffle=True, drop_last=True)
            val_loader = DataLoader(PairDataset(corpus.val), batch_size=cfg.train.batch_size, drop_last=True)
            best_val_loss, best_epoch = float('inf'), None
            for epoch in range(cfg.train.epochs):
                train_loss = trainer.train_epoch(loader)
                val_loss = trainer.eval_epoch(val_loader)
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss, best_epoch = val_loss, epoch
                    trainer.save()
                print(f'epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f}'
                      + (' (saved)' if improved else ''))
                patience = cfg.train.get('patience', 0)  # 0 disables early stopping
                if patience and best_epoch is not None and epoch - best_epoch >= patience:
                    print(f'Early stop: no val improvement for {patience} epochs')
                    break
            trainer.load()
            print(f'Loaded best checkpoint: epoch {best_epoch}, val loss {best_val_loss:.4f}')
            retriever.query_encoder.eval()
            retriever.document_encoder.eval()
            retriever.document_embedding = None
    else:
        retriever = TFIDF(documents=corpus.all_documents)

    ### Retrieve top-k documents and evaluate
    for name, split_queries in queries.items():
        relevancies = retrieveQueryAndGetRelevancies(retriever, split_queries, k=cfg.top_k)
        print(f'{name} results: {Evaluate(relevancies, split_queries)}')

if __name__ == '__main__':
    main()