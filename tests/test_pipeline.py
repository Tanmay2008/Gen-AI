import unittest
from text_extraction import extract_text_from_pdf
from vectorizer import vectorize_text
from retriever import Retriever
from rag_pipeline import RAGPipeline

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        """
        Set up initial conditions before each test case.
        """
        # Simulate PDF text data (instead of reading from actual PDF)
        self.sample_text_chunks = [
            "OpenStax is part of Rice University, which is a 501(c)(3)nonprofit charitable corporation",
            "In Concepts of Biology, most art contains attribution to its creator within the caption.",
            "All OpenStax textbooks undergo a rigorous review process."
        ]
        
        # Simulate embeddings (mock vectors) for the sample chunks (could also mock with actual model)
        self.sample_embeddings = vectorize_text(self.sample_text_chunks)
        
        # Initialize retriever and RAG pipeline with sample data
        self.retriever = Retriever(self.sample_embeddings, self.sample_text_chunks)
        self.rag_pipeline = RAGPipeline(self.sample_text_chunks)

    def test_text_extraction(self):
        """
        Test text extraction functionality from a PDF.
        """
        # Assuming we have a small sample PDF for testing
        pdf_path = "notebook/data/sample.pdf"
        extracted_text = extract_text_from_pdf(pdf_path, 15, 16)  # Extract from page 0 to 1
        
        self.assertIsInstance(extracted_text, list)
        self.assertGreater(len(extracted_text), 0)  # Ensure some text is extracted

    def test_vectorization(self):
        """
        Test the vectorization of text chunks.
        """
        sample_text = ["This is a test sentence for vectorization."]
        embeddings = vectorize_text(sample_text)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 1)  # One embedding per text chunk
        self.assertEqual(len(embeddings[0]), 768)  # Assuming 768-dimensional embeddings (BERT-like)

    def test_retriever(self):
        """
        Test that the retriever retrieves the correct text based on similarity.
        """
        query = "What is Coverage and Scope?"
        query_vector = vectorize_text([query])[0]
        
        retrieved_text = self.retriever.retrieve(query_vector, top_k=1)
        
        self.assertEqual(len(retrieved_text), 1)
        self.assertIn("Concepts of Biology textbook adheres", retrieved_text[0])  # Check if correct chunk is retrieved

    def test_rag_pipeline(self):
        """
        Test that the RAG pipeline generates an answer based on the retrieved context.
        """
        question = "What is Coverage and Scope?"
        generated_answer = self.rag_pipeline.generate_answer(question)
        
        self.assertIsInstance(generated_answer, str)
        self.assertIn("Concepts of Biology textbook adheres", generated_answer)  # Check that answer contains expected info

if __name__ == '__main__':
    unittest.main()