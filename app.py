import os
from text_extraction import extract_text_from_pdf
from vectorizer import vectorize_text
from retriever import Retriever
from rag_pipeline import RAGPipeline

def main():
    """
    Main function to execute the RAG pipeline for answering questions from the PDF content.
    """
    
    # Step 1: Extract Text from the PDF
    pdf_path = "notebook/data/ConceptsofBiology-WEB.pdf"  # Replace with the actual path to the PDF
    start_page = 14  # Define the range of pages to extract (start page of Chapter 1)
    end_page = 16    # Define the range of pages to extract (end page of Chapter 2)
    
    print("Extracting text from PDF...")
    text_chunks = extract_text_from_pdf(pdf_path, start_page, end_page)
    
    # Step 2: Vectorize the Extracted Text
    print("Vectorizing text chunks...")
    embeddings = vectorize_text(text_chunks)
    
    # Step 3: Initialize Retriever
    print("Initializing retriever...")
    retriever = Retriever(embeddings, text_chunks)
    
    # Step 4: Initialize RAG Pipeline with the text chunks
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(text_chunks)
    
    # Step 5: Input loop for querying
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ").strip()
        
        if question.lower() == 'exit':
            print("Exiting the application...")
            break
        
        # Step 6: Get the generated answer using RAG pipeline
        print("Retrieving and generating answer...")
        answer = rag_pipeline.generate_answer(question)
        print(f"\nGenerated Answer: {answer}")

if __name__ == "__main__":
    main()