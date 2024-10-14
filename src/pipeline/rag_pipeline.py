from transformers import T5ForConditionalGeneration, T5Tokenizer
from vectorizer import vectorize_text
from retriever import Retriever

class RAGPipeline:
    def __init__(self, text_chunks):
        """
        Initializes the RAG pipeline with retriever and generator.
        
        Args:
            text_chunks (List[str]): List of text chunks to use for retrieval.
        """
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.generator = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        self.embeddings = vectorize_text(text_chunks)
        self.retriever = Retriever(self.embeddings, text_chunks)

    def generate_answer(self, question):
        """
        Generates an answer for the given question using retrieval-augmented generation.
        
        Args:
            question (str): The input question.
        
        Returns:
            str: The generated answer.
        """
        query_vector = vectorize_text([question])[0]
        retrieved_chunks = self.retriever.retrieve(query_vector)
        context = " ".join(retrieved_chunks)
        
        input_text = f"question: {question} context: {context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.generator.generate(input_ids,max_new_tokens=400)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)