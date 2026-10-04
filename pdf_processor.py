
# Import PyMuPDF.
# We use "pymupdf" instead of the older "fitz" API.
import pymupdf

# Import the text splitter from LangChain.
# It will divide large PDF text into smaller chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list containing the text and page number 
        for every page that contains text.
    """

    # Open the PDF file.
    document = pymupdf.open(pdf_path)

    # This list will store the extracted pages.
    pages = []

    # Go through every page in the PDF.
    # enumerate(..., start=1) makes page numbers start from 1.
    for page_number, page in enumerate(document, start=1):

        # Extract text from the current page.
        text = page.get_text()

        # Remove unnecessary spaces from the beginning
        # and end of the extracted text.
        text = text.strip()

        # Only save pages that actually contain text.
        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    # Close the PDF after extraction is complete.
    document.close()

    # Return all extracted pages.
    return pages


def create_chunks(pages, source):
    """
    Split PDF text into smaller chunks.

    Args:
        pages: List of extracted PDF pages.
        source: Name/path of the original PDF.

    Returns:
        A list of chunks containing:
        - text
        - page number
        - source PDF
    """

    # Create a recursive text splitter.
    #
    # chunk_size = maximum approximate size of each chunk.
    # chunk_overlap = text shared between consecutive chunks.
    #
    # The overlap helps prevent important information from
    # being lost when a sentence/topic is split between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # This list will contain all chunks.
    chunks = []

    # Process each extracted page.
    for page in pages:

        # Split the page text into smaller chunks.
        page_chunks = text_splitter.split_text(page["text"])

        # Store every chunk.
        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": source
            })

    # Return all created chunks.
    return chunks



