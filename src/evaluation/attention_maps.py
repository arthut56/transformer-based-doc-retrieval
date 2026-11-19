import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_map(attention_map, queries_labels, keys_labels, print_values:bool=False, ax=None, color_bar:bool=True):
    if ax is None:
        fig, ax = plt.subplots(figsize = (10,6), dpi=300)
    else:
        fig = plt.gcf()
    im = ax.imshow(attention_map, cmap=sns.color_palette("viridis", as_cmap=True))
    ax.grid(False)
    ax.set_ylabel("$\mathbf{Q}$")
    ax.set_xlabel("$\mathbf{K}$")
    ax.set_yticks(np.arange(len(queries_labels)))
    ax.set_yticklabels(queries_labels)
    ax.set_xticks(np.arange(len(keys_labels)))
    ax.set_xticklabels(keys_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if print_values:
        for i in range(len(queries_labels)):
            for j in range(len(keys_labels)):
                text = ax.text(j, i, f"{attention_map[i, j]:.2f}",
                            ha="center", va="center", color="w")

    if color_bar:
      fig.colorbar(im, fraction=0.02, pad=0.04)
    fig.tight_layout()


@torch.no_grad()
def get_attention_maps(dpr, text: str, encoder: str = 'query'):
    if encoder == 'query':
        model, max_length = dpr.query_encoder, dpr.cfg.model.query_max_length
    else:
        model, max_length = dpr.document_encoder, dpr.cfg.model.doc_max_length
    model.eval()
    ids = dpr.tokenize([text], max_length).to(dpr.device)
    model(ids)
    tokens = dpr.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    maps = [layer_attn[0].cpu() for layer_attn in model.attention_maps()]
    return tokens, maps


def plot_all_heads(maps, tokens, title: str = ''):
    """Grid of heatmaps: one row per layer, one column per head."""
    n_layers, n_heads = len(maps), maps[0].size(0)
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers), dpi=150,
                             squeeze=False)
    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            ax.imshow(maps[layer][head].numpy(), cmap='viridis', vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
            ax.set_title(f'layer {layer} head {head}', fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_query_document_similarity(dpr, queries: list[str], documents: list[str]):
    dpr.query_encoder.eval()
    dpr.document_encoder.eval()
    q = dpr.encode(queries, dpr.query_encoder, dpr.cfg.model.query_max_length)
    d = dpr.encode(documents, dpr.document_encoder, dpr.cfg.model.doc_max_length)
    sim = (q @ d.T).softmax(dim=1).cpu().numpy()
    short = lambda s: s if len(s) <= 40 else s[:37] + '...'
    plot_attention_map(sim, [short(x) for x in queries], [short(x) for x in documents],print_values=True)
    return plt.gcf()
