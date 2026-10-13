# The exact architecture depends on the model, but this is the general process for a typical encoder-based Sentence Transformers model.
- Tokenizer: Converts text into tokens and maps them to integer IDs using a vocabulary.
- Embedding matrix: Maps each token ID to a learned initial vector.
- Transformer encoder: Uses the surrounding context to produce contextual token representations, which can then be pooled into a sentence embedding.

```
TEXT
"I love programming"
        |
        v
TOKENIZER
Splits text into tokens
        |
        v
TOKENS
["I", " love", " programming"]
        |
        v
VOCABULARY LOOKUP
Maps tokens to integer IDs
        |
        v
TOKEN IDs
[42, 381, 7291]
        |
        v
EMBEDDING MATRIX LOOKUP
Retrieves a learned vector for each ID
        |
        v
INITIAL TOKEN VECTORS
3 vectors × 384 dimensions
        |
        v
TRANSFORMER ENCODER
Uses attention and neural network layers
        |
        v
CONTEXTUAL TOKEN VECTORS
3 context-aware vectors
        |
        v
MEAN POOLING
Combines token vectors
        |
        v
SENTENCE EMBEDDING
1 vector × 384 dimensions
        |
        v
OPTIONAL NORMALIZATION
Scales the vector to unit length
```
