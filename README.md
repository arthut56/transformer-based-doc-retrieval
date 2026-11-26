# Dense Passage Retrieval with Transformer Encoders

02456 Deep Learning, project 4. TF-IDF and BM25 from scratch, a Transformer encoder and BERT trained as Dense Passage Retrieval (DPR) encoders, a stored vector index, and evaluation with P@k, R@k and MRR on FiQA and SQuAD.

## Setup

```bash
git clone git@github.com:arthut56/transformer-based-doc-retrieval.git
cd transformer-based-doc-retrieval
python3 -m venv .venv # on the DTU HPC, run `module load python3/3.12.4` first
source .venv/bin/activate
pip install -r requirements.txt
```

The datasets are downloaded to `retrieval_data/` on first use. The code runs on CUDA, Intel XPU or CPU, whichever is available.

## Usage

```bash
python main.py retriever=bm25 # BM25 on FiQA
python main.py retriever=tf_idf dataset=squad # TF-IDF on SQuAD
python main.py retriever=dpr do_train=true # train our Transformer encoder with DPR
python main.py retriever=dpr do_train=true model=bert # fine-tune BERT instead
python main.py retriever=dpr do_train=true model.pooling_type=cls train.epochs=10
python error_analysis.py --a <run>/per_query.json --b <run>/per_query.json
```

Settings live in `config/` (Hydra). Each run prints P@k, R@k and MRR for k = 1, 5, 10, 100 and writs its rankings to `per_query.json`. Trained runs save both encoders and the passage index (`doc_index.pt`) to `train.save_dir`, and later runs reuse the index.

## All experiments on the DTU HPC

`run_experiments.sh` runs every experiment in the report in one go: the baselines, each variant of our encoder (three repetitions), BERT, SQuAD and the attention maps. Every run gets its own folder in `results/`, and finished runs are skipped, so the script can simply be restarted.

```bash
bash run_experiments.sh prepare
tmux new -s dpr
a100sh
bash run_experiments.sh
```

Copy the results back to your machine. This skips the encoder weights except for the two checkpoints the notebook uses:

```bash
rsync -av --include='tweaks/warmup_170/rep1/*.pt' --include='bert/cls/rep1/*.pt' --exclude='*.pt' \
    <user>@login2.hpc.dtu.dk:<path>/transformer-based-doc-retrieval/results/ results/
```

## Notebook

`project_notebook.ipynb` shows how to use each part of the code. Sections 1–4 run without any experiment results. Sections 5–8 read `results/`. Figures and LaTeX tables are written to `results/figures/` and `results/tables/`.
