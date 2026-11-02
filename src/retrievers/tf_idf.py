from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix

class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted characters from text and lowercase

        Edge cases:
         "billionare-funded" -> "billionarefunded"
         "they'll" -> "theyll"
        """

        text = "".join(character for character in text if character.isalnum() or character == " ")
        text = text.lower()

        return text.split()

    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        sanitized_wordlist = self.PreprocessText(query)
        unique_wordlist_as_set = set(sanitized_wordlist)

        return list(unique_wordlist_as_set)

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """

        #Careful responsibilities
        return self.GetQueryVocabulary(document.GetText())

    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        document_frequencies = {}
        for document in self.index.GetDocuments():
            for term in self.GetDocumentVocabulary(document):
                document_frequencies[term] = document_frequencies.get(term, 0) + 1
        return document_frequencies

    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments())
        doc_freq = self.GetDocumentFrequencies()

        inv_document_frequencies = {}
        for term in self.corpus_vocabulary:
            inv_document_frequencies[term] = np.log(N/doc_freq[term])
        return inv_document_frequencies


    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        document_vocab = self.GetDocumentVocabulary(document)
        wordlist = self.PreprocessText(document.GetText())

        term_counts = {}
        for term in document_vocab:
            term_counts[term] = 0
            for word in wordlist:
                if word == term:
                    term_counts[term] += 1
        return term_counts

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        # TODO: Add tf-idf code
        IDF_dict = self.idf
        for row, document in enumerate(self.index.GetDocuments()):
            term_counts_dict = self.GetDocumentTermCounts(document)
            for term in term_counts_dict:
                specific_term_count = term_counts_dict[term]

                TF_term = specific_term_count / max(term_counts_dict.values())
                IDF_term = IDF_dict[term]

                TF_IDF_term = TF_term * IDF_term
                #TODO: put it in the matrix
                tfidf_matrix[row, self.term_to_index[term]] = TF_IDF_term

        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)
        print(f"Query vector shape: {query_vector.shape, query_vector.toarray()}")
        # TODO: add code
        for idx, query in enumerate(queries):
            for term in self.PreprocessText(query):
                if term in self.term_to_index:
                    query_vector[idx, self.term_to_index[term]] += 1
        return query_vector.tocsr()
            # go through all the words in the query
            # for a specific word, find the total occurences and find its index in the corpus vocab.
                #Idx row and idx of collumn.



    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    