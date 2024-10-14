import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings, text_chunks):
        """
        Initializes the retriever with FAISS and text chunks.
        
        Args:
            embeddings (np.array): Array of embeddings.
            text_chunks (List[str]): List of text chunks.
        """
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.text_chunks = text_chunks

    def retrieve(self, query_vector, top_k=3):
        """
        Retrieves the top-k most similar text chunks to the query.
        
        Args:
            query_vector (np.array): Vector for the query.
            top_k (int): Number of top results to return.
        
        Returns:
            List[str]: List of top-k text chunks.
        """
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        return [self.text_chunks[idx] for idx in indices[0]]