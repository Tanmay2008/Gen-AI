import pdfplumber

def extract_text_from_pdf(pdf_path, start_page, end_page):
    """
    Extracts text from a given range of pages in a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        start_page (int): Start page number.
        end_page (int): End page number.
    
    Returns:
        List[str]: List of text chunks from the PDF.
    """
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(start_page, end_page):
            page = pdf.pages[i]
            text_chunks.append(page.extract_text())
    return text_chunks