
# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# Import Ollama so we can communicate with the local LLM.
import ollama

# Import Path so we can display clean PDF filenames.
from pathlib import Path

# Import NumPy for working with embedding vectors.
import numpy as np

# Import FAISS so we can directly search the vector database.
import faiss

# Import pickle so we can load the saved document chunks.
import pickle


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

# The Ollama LLM used to generate answers.
LLM_MODEL = "qwen3:1.7b"

# The Ollama embedding model used when the FAISS database
# was originally created.
EMBEDDING_MODEL = "nomic-embed-text"

# Location of the FAISS index.
FAISS_INDEX_PATH = "faiss_index.bin"

# Location of the saved document chunks.
CHUNKS_PATH = "chunks.pkl"

# Number of chunks to retrieve.
TOP_K = 5

# Minimum similarity required before we allow the LLM
# to answer the question.
#
# Because your embeddings/index are normalized, FAISS
# inner-product scores can be treated as cosine similarity.
#
# We start with 0.35. We can adjust this later if needed.
SIMILARITY_THRESHOLD = 0.35


# ---------------------------------------------------------
# LOAD FAISS DATABASE
# ---------------------------------------------------------

# Load the FAISS vector index.
index = faiss.read_index(FAISS_INDEX_PATH)


# ---------------------------------------------------------
# LOAD DOCUMENT CHUNKS
# ---------------------------------------------------------

# Open the chunks file.
with open(CHUNKS_PATH, "rb") as file:

    # Load all saved chunks.
    chunks = pickle.load(file)


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

# Strong instructions for the LLM.
SYSTEM_PROMPT = """
You are a strict PDF RAG assistant.

You MUST answer the user's question ONLY using the
information provided in the context.

Do NOT use your own general knowledge.

Do NOT guess.

Do NOT add information that is not present in the context.

If the context does not contain enough information to
answer the question, respond exactly with:

"I could not find the answer in the provided documents."

Keep the answer clear and concise.
"""


# ---------------------------------------------------------
# CREATE QUERY EMBEDDING
# ---------------------------------------------------------

def create_query_embedding(query):
    """
    Convert the user's question into an embedding
    using the SAME model used to create the FAISS database.
    """

    # Create an embedding using Ollama.
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    # Get the first embedding from the response.
    embedding = response["embeddings"][0]

    # Convert the embedding to float32.
    return np.array(
        embedding,
        dtype="float32"
    )


# ---------------------------------------------------------
# RETRIEVE DOCUMENTS
# ---------------------------------------------------------

def retrieve_documents(query, top_k=TOP_K):
    """
    Search FAISS for chunks relevant to the user's question.

    Returns:
        relevant_documents
        similarity_scores
    """

    # Create an embedding for the user's question.
    query_embedding = create_query_embedding(query)

    # FAISS expects a 2-dimensional NumPy array.
    query_vector = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Search the FAISS index.
    distances, indices = index.search(
        query_vector,
        top_k
    )

    # Store relevant documents here.
    relevant_documents = []

    # Store similarity scores here.
    similarity_scores = []

    # Go through all retrieved results.
    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):

        # Ignore invalid FAISS indexes.
        if index_position == -1:
            continue

        # Check whether the similarity is high enough.
        if distance >= SIMILARITY_THRESHOLD:

            # Get the corresponding document chunk.
            document = chunks[index_position]

            # Save the document.
            relevant_documents.append(document)

            # Save its similarity score.
            similarity_scores.append(float(distance))

    # Return both documents and scores.
    return relevant_documents, similarity_scores


# ---------------------------------------------------------
# BUILD CONTEXT
# ---------------------------------------------------------

def build_context(documents):
    """
    Convert retrieved document chunks into context
    that will be given to the LLM.
    """

    # Store individual context sections.
    context_parts = []

    # Process each document.
    for document in documents:

        # Get the text of the chunk.
        text = document["text"]

        # Get the page number.
        page = document["page"]

        # Get the source path.
        source = document["source"]

        # Get only the PDF filename.
        filename = Path(source).name

        # Add source and page information.
        context_parts.append(
            f"[Source: {filename}, Page: {page}]\n"
            f"{text}"
        )

    # Combine all chunks into one context.
    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# ASK RAG
# ---------------------------------------------------------

def ask_rag(question, top_k=TOP_K):
    """
    Complete RAG pipeline:

    Question
        ↓
    Query embedding
        ↓
    FAISS search
        ↓
    Similarity filtering
        ↓
    Context
        ↓
    Qwen
        ↓
    Answer + citations
    """

    # -----------------------------------------------------
    # STEP 1: RETRIEVE DOCUMENTS
    # -----------------------------------------------------

    documents, scores = retrieve_documents(
        question,
        top_k
    )


    # -----------------------------------------------------
    # STEP 2: CHECK WHETHER RELEVANT INFORMATION EXISTS
    # -----------------------------------------------------

    # If no document passed the similarity threshold,
    # we do NOT call the LLM.
    if not documents:

        return (
            "I could not find the answer in the provided documents.",
            []
        )


    # -----------------------------------------------------
    # STEP 3: BUILD CONTEXT
    # -----------------------------------------------------

    # Convert the retrieved chunks into context.
    context = build_context(documents)


    # -----------------------------------------------------
    # STEP 4: CREATE PROMPT
    # -----------------------------------------------------

    user_prompt = f"""
Context from the provided PDF documents:

{context}

User Question:
{question}

Answer the question using ONLY the context above.

If the answer is not contained in the context, say:

"I could not find the answer in the provided documents."
"""


    # -----------------------------------------------------
    # STEP 5: ASK OLLAMA
    # -----------------------------------------------------

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )


    # -----------------------------------------------------
    # STEP 6: GET ANSWER
    # -----------------------------------------------------

    answer = response["message"]["content"].strip()


    # -----------------------------------------------------
    # STEP 7: CHECK FOR "NOT FOUND"
    # -----------------------------------------------------

    # If the LLM says it could not find the answer,
    # do not show irrelevant citations.
    if (
        "I could not find the answer"
        in answer
    ):

        return (
            "I could not find the answer in the provided documents.",
            []
        )


    # -----------------------------------------------------
    # STEP 8: CREATE CITATIONS
    # -----------------------------------------------------

    citations = []

    # Prevent duplicate source/page combinations.
    seen = set()

    # Process each retrieved document.
    for document in documents:

        # Get source path.
        source = document["source"]

        # Get page number.
        page = document["page"]

        # Get only filename.
        filename = Path(source).name

        # Create unique citation.
        citation = (
            filename,
            page
        )

        # Add only unique citations.
        if citation not in seen:

            citations.append(citation)

            seen.add(citation)


    # -----------------------------------------------------
    # STEP 9: RETURN ANSWER + CITATIONS
    # -----------------------------------------------------

    return answer, citations


# ---------------------------------------------------------
# COMMAND-LINE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nPDF RAG Chatbot")
    print("----------------------------")
    print("Ask questions about your PDF.")
    print("Type 'exit' to stop.\n")


    # Keep the chatbot running.
    while True:

        # Ask the user for a question.
        question = input("You: ").strip()


        # Stop the program.
        if question.lower() == "exit":

            print("Goodbye!")

            break


        # Ignore empty questions.
        if not question:
            continue


        # Ask the RAG system.
        answer, citations = ask_rag(question)


        # Display the answer.
        print("\nAI:")
        print(answer)


        # Display citations only when available.
        if citations:

            print("\nSources:")

            for source, page in citations:

                print(
                    f"- {source} — Page {page}"
                )


        # Separator.
        print(
            "\n" + "-" * 60 + "\n"
        )

