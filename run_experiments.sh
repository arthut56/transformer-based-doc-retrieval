#!/bin/bash

cd "${LS_SUBCWD:-$(dirname "$0")}"
module load python3/3.12.4
source .venv/bin/activate

RESULTS=results
REPEATS="1 2 3"
mkdir -p "$RESULTS"

if [ "$1" == "prepare" ]; then
    python -c "
import datasets, transformers
transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
transformers.AutoModel.from_pretrained('bert-base-uncased')
from src.data_utils.corpus import Corpus
Corpus(dataset_name='fiqa', base_dir='./retrieval_data')
Corpus(dataset_name='squad', base_dir='./retrieval_data')
print('Tokenizer and datasets cached.')
"
    exit
fi

export HF_HUB_OFFLINE=1

run() {
    local name=$1; shift
    local dir="$RESULTS/$name"
    if grep -q "Val results" "$dir/output.log" 2>/dev/null; then
        echo "[skip, done] $name"
        return
    fi
    mkdir -p "$dir"
    echo "[run] $name: $*"
    python main.py "$@" hydra.run.dir="$dir" train.save_dir="$dir" 2>&1 | tee "$dir/output.log"
}

run_dpr() {
    local name=$1; shift
    for rep in $REPEATS; do
        run "${name}/rep${rep}" retriever=dpr do_train=true "$@"
    done
}

run baselines/tf_idf_fiqa retriever=tf_idf dataset=fiqa
run baselines/bm25_fiqa retriever=bm25 dataset=fiqa
run baselines/tf_idf_squad retriever=tf_idf dataset=squad
run baselines/bm25_squad retriever=bm25 dataset=squad

run_dpr pooling/cls model.pooling_type=cls
run_dpr pooling/mean model.pooling_type=mean
run_dpr pooling/max model.pooling_type=max

run_dpr regularisation/dropout_0.3 model.dropout=0.3
run_dpr regularisation/weight_decay_0.1 train.weight_decay=0.1
run_dpr regularisation/layers_2 model.n_layers=2

run_dpr tweaks/warmup_170 +train.warmup_steps=170
run_dpr tweaks/pre_norm +model.norm_first=true
run_dpr tweaks/hidden_128 model.hidden_dim=128

run_dpr batch_size/32 train.batch_size=32
run_dpr batch_size/128 train.batch_size=128

run squad/dpr_mean retriever=dpr do_train=true dataset=squad

BERT="model=bert train.lr=0.00002 train.batch_size=32 train.epochs=10 train.patience=3 +train.warmup_steps=171"
run bert/no_finetune retriever=dpr do_train=false model=bert
run_dpr bert/cls $BERT model.pooling_type=cls
run_dpr bert/mean $BERT model.pooling_type=mean
run squad/bert_cls retriever=dpr do_train=true dataset=squad $BERT model.pooling_type=cls

if [ ! -f "$RESULTS/attention/trained_query_heads.png" ];  then
    echo "[run] attention maps"
    python visualize_attention.py +attention.weights_dir="$RESULTS/pooling/mean/rep1" \
        +attention.out_dir="$RESULTS/attention" hydra.run.dir="$RESULTS/attention" 2>&1 | tee "$RESULTS/attention_output.log"
fi

if [ ! -f "$RESULTS/attention_bert/trained_query_heads.png" ]; then
    echo "[run] BERT attention maps"
    python visualize_attention.py model=bert +attention.weights_dir="$RESULTS/bert/cls/rep1" \
        +attention.out_dir="$RESULTS/attention_bert" hydra.run.dir="$RESULTS/attention_bert" 2>&1 | tee "$RESULTS/attention_bert_output.log"
fi

grep -r --include=output.log -E "results:|Loaded best checkpoint" "$RESULTS" | sort > "$RESULTS/summary.txt"
echo "Done, you'll find a summary in $RESULTS/summary.txt"
