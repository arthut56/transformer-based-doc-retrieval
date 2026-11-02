from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b

        self.bm25_matrix = self.GetDocumentsTFIDFVectors()

    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments()) + 1
        doc_freq = self.GetDocumentFrequencies()

        inv_document_frequencies = {}
        for term in self.corpus_vocabulary:
            inv_document_frequencies[term] = np.log(N/(doc_freq[term]+0.5))
        return inv_document_frequencies

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments()) # N
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        doc_lengths, d_avg = self.GetDocumentLengths()

        IDF_dict = self.idf
        for row, document in enumerate(self.index.GetDocuments()):
            term_counts_dict = self.GetDocumentTermCounts(document)


            for term in term_counts_dict:
                specific_term_count = term_counts_dict[term]

                TF_term = (specific_term_count * (self.k1 + 1)) / (specific_term_count + self.k1 * (1 - self.b + self.b * (doc_lengths[document] / d_avg)))
                IDF_term = IDF_dict[term]

                TF_IDF_term = TF_term * IDF_term
                bm25_matrix[row, self.term_to_index[term]] = TF_IDF_term

        return bm25_matrix.tocsr()   # Convert to CSR format for efficient row slicin

    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores