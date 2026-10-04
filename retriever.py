
# ============================================================
# retriever.py
# ============================================================
# This file:
#
# 1. Loads the FAISS database
# 2. Loads the saved PDF chunks
# 3. Creates an embedding for the user's question
# 4. Searches FAISS
# 5. Returns matching chunks AND their L2 distances
#
# IMPORTANT:
# Our FAISS index uses IndexFlatL2.
#
# Therefore:
# Smaller L2 distance = MORE similar
# Larger L2 distance = LESS similar
# ============================================================


import numpy as np
import pickle
import faiss
import ollama


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL = "nomic-embed-text"


# ------------------------------------------------------------
# Load FAISS index
# ------------------------------------------------------------

index = faiss.read_index(FAISS_INDEX_PATH)


# ------------------------------------------------------------
# Load saved chunks
# ------------------------------------------------------------

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)


# ------------------------------------------------------------
# Display database information
# ------------------------------------------------------------

print("FAISS retriever loaded successfully!")
print(f"Number of vectors: {index.ntotal}")
print(f"Embedding dimension: {index.d}")
print(f"Number of chunks: {len(chunks)}")


# ============================================================
# Create embedding for user question
# ============================================================

def create_query_embedding(query):
    """
    Convert the user's question into an embedding.
    """

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    embedding = response["embeddings"][0]

    return np.array(
        embedding,
        dtype="float32"
    )


# ============================================================
# Retrieve documents
# ============================================================

def retrieve_documents(query, top_k=5):
    """
    Search FAISS for the most similar PDF chunks.

    Returns:
        retrieved_chunks
        retrieved_distances

    With IndexFlatL2:

        Smaller distance = more similar
        Larger distance = less similar
    """

    # Create embedding for the question
    query_embedding = create_query_embedding(query)


    # FAISS expects a 2D array:
    #
    # Shape:
    # (1, 768)
    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )


    # Search FAISS
    distances, indices = index.search(
        query_vector,
        top_k
    )


    # Lists to store results
    retrieved_chunks = []
    retrieved_distances = []


    # Process results
    for distance, chunk_index in zip(
        distances[0],
        indices[0]
    ):

        # FAISS can return -1 when no result exists
        if chunk_index == -1:
            continue


        # Get the corresponding PDF chunk
        chunk = chunks[chunk_index]


        # Save chunk
        retrieved_chunks.append(chunk)


        # Save L2 distance
        retrieved_distances.append(
            float(distance)
        )


    return retrieved_chunks, retrieved_distances


# ============================================================
# Test the retriever
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    ).strip()


    if query:

        # Retrieve documents
        results, distances = retrieve_documents(
            query,
            top_k=5
        )


        print("\n" + "=" * 60)
        print("RETRIEVED DOCUMENTS")
        print("=" * 60)


        # Display every result
        for i, (result, distance) in enumerate(
            zip(results, distances),
            start=1
        ):

            print(
                f"\n--- Result {i} ---"
            )

            print(
                f"L2 Distance: {distance:.4f}"
            )

            print(result)


    else:

        print(
            "Please enter a question."
        )

