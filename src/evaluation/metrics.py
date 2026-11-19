from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculateStandardRecall(relevancies: list[bool], query: Query):
    """Recall@k: hits / number of relevant documents."""
    return sum(relevancies) / query.getNumberOfRelevantDocuments()

def calculatePrecision(relevancies: list[bool]):
    return sum(relevancies) / len(relevancies)

def calculateReciprocalRank(relevancies: list[bool]):
    for rank, hit in enumerate(relevancies, start=1):
        if hit:
            return 1 / rank
    return 0.0

def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    n_queries = len(queries)
    totals = {'recall': 0.0, 'recall_standard': 0.0, 'precision': 0.0, 'mrr': 0.0}

    for rel, query in zip(relevancies, queries):
        totals['recall'] += calculateRecall(rel, query)
        totals['recall_standard'] += calculateStandardRecall(rel, query)
        totals['precision'] += calculatePrecision(rel)
        totals['mrr'] += calculateReciprocalRank(rel)

    return {name: total / n_queries for name, total in totals.items()}
