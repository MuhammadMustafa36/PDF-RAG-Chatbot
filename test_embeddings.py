
# Import the Ollama Python library
import ollama


# Text that we want to convert into an embedding
text = "Machine learning allows computers to learn from data."


# Generate an embedding using the Ollama embedding model
response = ollama.embed(
    model="nomic-embed-text",
    input=text
)


# Get the embedding vector
embedding = response["embeddings"][0]


# Display information about the embedding
print("Embedding generated successfully!")

# Number of values in the vector
print(f"Vector dimensions: {len(embedding)}")

# Display the first 10 values
print("\nFirst 10 values:")
print(embedding[:10])

