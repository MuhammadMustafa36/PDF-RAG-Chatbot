# first we import data  in test_pdf file 
# create two functions one for extracting data and another for creating chunks of data in pdf_processor file
# then import the functions in test_pdf file and use them to extract data from pdf and create chunks of data from it
# Import our PDF processing functions
from pdf_processor import extract_text_from_pdf, create_chunks


# PDF location
pdf_path = "data/Machine_learning_book.pdf"


# Step 1: Extract PDF pages
pages = extract_text_from_pdf(pdf_path)

print(f"Total pages: {len(pages)}")


# Step 2: Create chunks
# Pass the PDF path as the source
chunks = create_chunks(pages, pdf_path)

print(f"Total chunks: {len(chunks)}")


# Display the first 3 chunks
print("\nFirst 3 chunks:\n")


for i, chunk in enumerate(chunks[:3], start=1):

    print(f"--- Chunk {i} ---")

    # Display the page number
    print(f"Page: {chunk['page']}")

    # Display the source PDF
    print(f"Source: {chunk['source']}")

    # Display the chunk text
    print(chunk["text"])

    print()
