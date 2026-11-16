import hydra

from src.retrievers.bm25 import BM25
from src.data_utils.corpus import Corpus
from src.retrievers.tf_idf import TFIDF
from src.data_utils.query import Query
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate
from src.retrievers.dpr import DPR
from torch.utils.data import DataLoader
from src.training.trainer import Trainer, PairDataset

@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    ### Prepare queries
    train_queries = [Query(corpus.train['question'][idx], 
                           id=corpus.train['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.train['query_id'][idx]]
                           )
                        for idx in range(corpus.train.num_rows)
                    ]

    val_queries =   [Query(corpus.val['question'][idx], 
                         id=corpus.val['query_id'][idx], 
                         relevant_document_ids=corpus.relevant_ids[corpus.val['query_id'][idx]]
                           ) 
                        for idx in range(corpus.val.num_rows)
                    ]
    
    test_queries =  [Query(corpus.test['question'][idx], 
                           id=corpus.test['query_id'][idx], 
                           relevant_document_ids=corpus.relevant_ids[corpus.test['query_id'][idx]]
                           ) 
                        for idx in range(corpus.test.num_rows)
                    ]

    if cfg.retriever.method == 'bm25':
        retriever = BM25(documents=corpus.all_documents)
    elif cfg.retriever.method == 'dpr':
        retriever = DPR(cfg, documents=corpus.all_documents)
        if cfg.get('do_train', False):
            trainer = Trainer(retriever, cfg)
            loader = DataLoader(PairDataset(corpus.train), batch_size=cfg.train.batch_size,
                                shuffle=True, drop_last=True)
            val_loader = DataLoader(PairDataset(corpus.val), batch_size=cfg.train.batch_size, drop_last=True)
            for epoch in range(cfg.train.epochs):
                train_loss = trainer.train_epoch(loader)
                val_loss = trainer.eval_epoch(val_loader)
                print(f'epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f}')
            trainer.save()
            retriever.query_encoder.eval()
            retriever.document_encoder.eval()
            retriever.document_embedding = None  # stale after training; rebuilt on next lookup
    else:
        retriever = TFIDF(documents=corpus.all_documents)

    ### Retrieve top-k relevant queries
    train_relevancies = retrieveQueryAndGetRelevancies(retriever, train_queries, k=cfg.top_k)
    val_relevancies = retrieveQueryAndGetRelevancies(retriever, val_queries, k=cfg.top_k)
    test_relevancies = retrieveQueryAndGetRelevancies(retriever, test_queries, k=cfg.top_k)

    ### Evaluate
    train_scores = Evaluate(train_relevancies, train_queries)
    val_scores = Evaluate(val_relevancies, val_queries)
    test_scores = Evaluate(test_relevancies, test_queries)

    print(f'Train results: {train_scores}')
    print(f'Val results: {val_scores}')
    print(f'Test results: {test_scores}')

if __name__ == '__main__':
    main()