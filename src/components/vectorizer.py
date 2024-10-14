from sentence_transformers import SentenceTransformer

def vectorize_text(text_chunks):
    """
    Converts text chunks into vectors using a pre-trained model.
    
    Args:
        text_chunks (List[str]): List of text chunks.
    
    Returns:
        List[np.array]: List of vectors corresponding to text chunks.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    return embeddings