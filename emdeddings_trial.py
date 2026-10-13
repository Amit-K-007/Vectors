from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.get_embedding_dimension())   # 384
print(model.max_seq_length)              # 256 tokens

sentences = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "Best pizza places in Mumbai",
    "The server is up",
    "The server is not up",
]
vecs = model.encode(sentences, normalize_embeddings=True)
print(vecs.shape)  # (5, 384)

vecs_transpose = vecs.T

# Matrix multiplication
sim = vecs @ vecs_transpose
np.set_printoptions(precision=2, suppress=True)
print(sim)


#
text = "This is a sample sentence used to test token truncation. " * 100
print("Maximum sequence length:", model.max_seq_length)

tokens = model.tokenizer(text, truncation=False)["input_ids"]
print("Original token count:", len(tokens))

# Check token count after truncation
tokens_after = model.tokenizer(
    text,
    truncation=True,
    max_length=model.max_seq_length
)["input_ids"]
print("Tokens after truncation:", len(tokens_after))


#
text = "I love programming and building applications."

tokens = model.tokenizer.tokenize(text)

print("Tokens:", tokens)
print("Token count:", len(tokens))
