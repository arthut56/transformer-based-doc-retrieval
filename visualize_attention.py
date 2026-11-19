"""
Save attention-map figures for an untrained and a trained DPR query/document encoder.

    python visualize_attention.py                          # uses outputs/dpr/*.pt as trained weights
    python visualize_attention.py +attention.text="..."    # custom query text

Figures go to outputs/attention/.
"""
import os
import hydra
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_utils.corpus import Corpus
from src.retrievers.dpr import DPR
from src.evaluation.attention_maps import (get_attention_maps, plot_all_heads,
                                           plot_query_document_similarity)

DEFAULT_TEXT = 'how do I deduct business travel expenses'


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    att_cfg = cfg.get('attention', {})
    text = att_cfg.get('text', DEFAULT_TEXT)
    out_dir = att_cfg.get('out_dir', 'outputs/attention')
    weights_dir = att_cfg.get('weights_dir', cfg.train.save_dir)
    os.makedirs(out_dir, exist_ok=True)

    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')
    dpr = DPR(cfg, documents=corpus.all_documents)

    # A few validation queries with one of their relevant documents, for the similarity map
    val = corpus.val
    sample_queries = list(val['question'][:6])
    sample_docs = [texts[0] for texts in val['text'][:6]]

    def save_all(tag):
        for encoder in ('query', 'document'):
            tokens, maps = get_attention_maps(dpr, text, encoder=encoder)
            fig = plot_all_heads(maps, tokens, title=f'{tag} {encoder} encoder: "{text}"')
            fig.savefig(f'{out_dir}/{tag}_{encoder}_heads.png', bbox_inches='tight')
            plt.close(fig)
        fig = plot_query_document_similarity(dpr, sample_queries, sample_docs)
        fig.savefig(f'{out_dir}/{tag}_query_doc_similarity.png', bbox_inches='tight')
        plt.close(fig)

    save_all('untrained')

    q_path, d_path = f'{weights_dir}/query_encoder.pt', f'{weights_dir}/document_encoder.pt'
    if os.path.exists(q_path) and os.path.exists(d_path):
        dpr.query_encoder.load_state_dict(torch.load(q_path, map_location=dpr.device))
        dpr.document_encoder.load_state_dict(torch.load(d_path, map_location=dpr.device))
        save_all('trained')
    else:
        print(f'No trained weights in {weights_dir}; saved untrained maps only.')

    print(f'Saved figures to {out_dir}/')


if __name__ == '__main__':
    main()
