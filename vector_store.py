# Import the Ollama Python library.
# We will use it to generate embeddings.
import ollama

# Import FAISS.
# FAISS will store and search our vectors efficiently.
import faiss

# Import NumPy.
# FAISS requires vectors in NumPy array format.
import numpy as np

# Import pickle.
# We will use it to save our chunk information.
import pickle

# Import our PDF processing functions.
from pdf_processor import extract_text_from_pdf, create_chunks


# ---------------------------------------------------------
# 1. PDF PATH
# ---------------------------------------------------------

# Location of our PDF file.
pdf_path = "data/Machine_learning_book.pdf"


# ---------------------------------------------------------
# 2. EXTRACT PDF TEXT
# ---------------------------------------------------------

# Extract text from all readable PDF pages.
pages = extract_text_from_pdf(pdf_path)

print(f"Total readable pages: {len(pages)}")


# ---------------------------------------------------------
# 3. CREATE CHUNKS
# ---------------------------------------------------------

# Divide the PDF text into smaller pieces.
chunks = create_chunks(pages, pdf_path)

print(f"Total chunks: {len(chunks)}")


# ---------------------------------------------------------
# 4. GENERATE EMBEDDINGS
# ---------------------------------------------------------

# This list will contain the embedding vector
# for every chunk.
embeddings = []


# Process every chunk.
for i, chunk in enumerate(chunks):

    print(f"Creating embedding {i + 1}/{len(chunks)}")

    # Generate an embedding using Ollama.
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk["text"]
    )

    # Get the actual 768-dimensional vector.
    embedding = response["embeddings"][0]

    # Add it to our embeddings list.
    embeddings.append(embedding)


# ---------------------------------------------------------
# 5. CONVERT TO NUMPY ARRAY
# ---------------------------------------------------------

# Convert our Python list into a NumPy array.
#
# FAISS works with NumPy arrays containing float32 values.
embeddings = np.array(embeddings, dtype="float32")


# Display the shape of our embeddings.
print("\nEmbeddings created successfully!")

print(f"Embedding shape: {embeddings.shape}")


# ---------------------------------------------------------
# 6. CREATE FAISS INDEX
# ---------------------------------------------------------

# Get the number of dimensions.
#
# For nomic-embed-text this should be 768.
dimension = embeddings.shape[1]


# Create a FAISS index using L2 distance.

# L2 distance measures how close two vectors are.
index = faiss.IndexFlatL2(dimension)


# Add all embeddings to the FAISS index.
index.add(embeddings)


# Display the number of vectors stored.
print(f"Vectors stored in FAISS: {index.ntotal}")


# ---------------------------------------------------------
# 7. SAVE FAISS INDEX
# ---------------------------------------------------------

# Save the FAISS index to disk.
faiss.write_index(index, "faiss_index.bin")


# ---------------------------------------------------------
# 8. SAVE CHUNK INFORMATION
# ---------------------------------------------------------

# Save the original chunks separately.
#
# FAISS stores vectors, but it does NOT store our
# original text, page numbers, or source information.
with open("chunks.pkl", "wb") as file:

    pickle.dump(chunks, file)


# ---------------------------------------------------------
# 9. FINISHED
# ---------------------------------------------------------

print("\nVector database created successfully!")

print("Saved files:")
print("- faiss_index.bin")
print("- chunks.pkl")