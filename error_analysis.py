"""
Example Usage:
    python error_analysis.py --a results/baselines/bm25_fiqa/rescore/per_query.json --a_name BM25 \
        --b results/bert/cls/rep1/rescore/per_query.json --b_name BERT --out results/error_analysis.md
"""
import argparse
import json
from src.data_utils.corpus import Corpus


def first_hit(entry):
    """1-based rank of the first relevant passage, or None if none was retrieved."""
    relevant = set(entry['relevant'])
    for rank, doc_id in enumerate(entry['retrieved'], start=1):
        if doc_id in relevant:
            return rank
    return None


def shorten(text, n=300):
    text = ' '.join(text.split())
    return text if len(text) <= n else text[:n] + '...'


def describe(entry, rank, name, doc_text, k):
    found = f'first relevant at rank {rank}' if rank else f'no relevant passage in top {len(entry["retrieved"])}'
    status = 'hit' if rank and rank <= k else 'miss'
    if rank == 1:
        return [f'**{name}** ({status}, {found}): its top passage is the relevant one.', '']
    return [f'**{name}** ({status}, {found}). Top passage: "{shorten(doc_text[entry["retrieved"][0]])}"', '']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--a_name', default='A')
    parser.add_argument('--b_name', default='B')
    parser.add_argument('--dataset', default='fiqa')
    parser.add_argument('--split', default='Test')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--n', type=int, default=5, help='examples shown per direction')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    a = {e['id']: e for e in json.load(open(args.a))[args.split]}
    b = {e['id']: e for e in json.load(open(args.b))[args.split]}
    assert a.keys() == b.keys(), 'The two files must cover the same queries'

    corpus = Corpus(dataset_name=args.dataset, base_dir='./retrieval_data')
    docs = corpus.all_documents
    doc_text = {ids[0]: texts[0] for texts, ids in zip(docs['text'], docs['document_id'])}

    hits = {qid: (first_hit(a[qid]), first_hit(b[qid])) for qid in a}
    in_k = lambda rank: rank is not None and rank <= args.k
    both = [q for q, (ra, rb) in hits.items() if in_k(ra) and in_k(rb)]
    only_a = [q for q, (ra, rb) in hits.items() if in_k(ra) and not in_k(rb)]
    only_b = [q for q, (ra, rb) in hits.items() if in_k(rb) and not in_k(ra)]
    neither = [q for q, (ra, rb) in hits.items() if not in_k(ra) and not in_k(rb)]

    lines = [f'# {args.a_name} vs {args.b_name} ({args.dataset}, {args.split}, top {args.k})', '',
             f'| | queries |', '|---|---|',
             f'| both find a relevant passage | {len(both)} |',
             f'| only {args.a_name} | {len(only_a)} |',
             f'| only {args.b_name} | {len(only_b)} |',
             f'| neither | {len(neither)} |',
             f'| total | {len(hits)} |', '']

    # show the clearest cases first. The winner ranks a relevant passage highest
    for winner, loser, wins, w_name, l_name, w_idx in [
            (a, b, only_a, args.a_name, args.b_name, 0),
            (b, a, only_b, args.b_name, args.a_name, 1)]:
        lines += [f'## {w_name} finds it, {l_name} does not', '']
        for qid in sorted(wins, key=lambda q: hits[q][w_idx])[:args.n]:
            entry = winner[qid]
            relevant = entry['retrieved'][hits[qid][w_idx] - 1]  # the relevant passage the winner found
            lines += [f'### Query: "{entry["query"]}"', '',
                      f'Relevant passage: "{shorten(doc_text[relevant])}"', '']
            lines += describe(entry, hits[qid][w_idx], w_name, doc_text, args.k)
            lines += describe(loser[qid], hits[qid][1 - w_idx], l_name, doc_text, args.k)

    report = '\n'.join(lines)
    print(report)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(report)
        print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
