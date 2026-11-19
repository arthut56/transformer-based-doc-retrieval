# 02456 Retrieval Project

Starter code for information retrieval project work in the 02456 Deep Learning course (Fall 2026).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py retriever=tf_idf                 # TF-IDF on FiQA
python main.py retriever=bm25                   # BM25 on FiQA
python main.py retriever=dpr do_train=true      # train DPR encoders, then evaluate the best checkpoint
python main.py retriever=bm25 dataset=squad     # SQuAD (no test split; train and val are evaluated)
python main.py top_k=20                         # Override a config value
python visualize_attention.py                   # attention maps, untrained vs trained, to outputs/attention/
```

This prints recall@k (capped at min(k, number relevant)), standard recall@k, precision@k and MRR@k for each split. Settings are managed with Hydra in `config/`.

## Structure

```
config/           Experiment, dataset, retriever, and model settings.
main.py           Runs retrieval using TF-IDF and a preliminary evaluation pipeline.
src/
├── data_utils/   Dataset loading, corpus, document and query classes.
├── retrievers/   Retriever base class, TF-IDF, BM25 and DPR.
├── training/     DPR loss and trainer.
└── evaluation/   Recall metric.
```