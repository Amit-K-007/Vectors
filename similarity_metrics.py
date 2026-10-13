import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
sents = [
    "How do I reset my password?",
    "I forgot my login credentials",
    "Best pizza places in Mumbai"
]

raw = model.encode(sents, normalize_embeddings=False)
print("Norms (raw original):", np.linalg.norm(raw, axis=1))

# ---------------------------------------------------------
# 1. Base computations (Original)
# ---------------------------------------------------------
unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
q = unit[0]

cos_orig = unit @ q
dot_raw_orig = raw @ raw[0]
l2_orig = np.linalg.norm(unit - q, axis=1)

print("\n--- Original Metrics ---")
print("Cosine (Original): ", cos_orig)
print("Dot (Raw Original):", dot_raw_orig)
print("L2 (Original):     ", l2_orig)
breakpoint()
# ---------------------------------------------------------
# 2. Deliberately scale the first vector by 10
# ---------------------------------------------------------
raw_scaled = raw.copy()
raw_scaled[0] = raw_scaled[0] * 10

print("\nNorms (raw scaled):  ", np.linalg.norm(raw_scaled, axis=1))

# Re-compute unit vectors for the scaled data
unit_scaled = raw_scaled / np.linalg.norm(raw_scaled, axis=1, keepdims=True)
q_scaled = unit_scaled[0]

# Metrics with the scaled vector
cos_scaled = unit_scaled @ q_scaled
dot_raw_scaled = raw_scaled @ raw_scaled[0]
l2_scaled = np.linalg.norm(unit_scaled - q_scaled, axis=1)

# L2 distance on the UNNORMALIZED raw scaled vectors to see magnitude impact
l2_raw_scaled = np.linalg.norm(raw_scaled - raw_scaled[0], axis=1)

print("\n--- Scaled Metrics ---")
print("Cosine (Scaled):         ", cos_scaled)
print("Dot (Raw Scaled):        ", dot_raw_scaled)
print("L2 (Normalized Scaled):  ", l2_scaled)
print("L2 (Raw Unnorm Scaled):  ", l2_raw_scaled)
print("Check identity formula:", np.allclose(l2_scaled**2, 2 - 2 * cos_scaled))
