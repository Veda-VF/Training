"""
flowchart
    A([Start]) --> B[Create 6 token embeddings]
    B --> C[Generate sinusoidal positional encoding]
    C --> D[Add position information to embeddings]
    B --> E[Calculate attention without position]
    D --> F[Calculate attention with position]
    E --> G[Print both attention matrices]
    F --> G
    G --> H([End])
"""
import numpy as np

# Generate the same random numbers every time we run the program.
np.random.seed(42)


# Convert scores into weights that add up to 1.
def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


# Generate sinusoidal position information for each token.
def positional_encoding(seq_len, d_model):
    encoding = np.zeros((seq_len, d_model))

    # Token positions: 0, 1, 2, 3, ...
    positions = np.arange(seq_len)[:, np.newaxis]

    # Different dimensions use different frequencies.
    frequencies = 10000 ** (
        np.arange(0, d_model, 2) / d_model
    )

    # Even columns use sine.
    encoding[:, 0::2] = np.sin(positions / frequencies)

    # Odd columns use cosine.
    # Slice frequencies to support odd model dimensions too.
    encoding[:, 1::2] = np.cos(
        positions / frequencies[:encoding[:, 1::2].shape[1]]
    )

    return encoding


# Calculate self-attention and return its attention weights.
def self_attention(X):
    # In this simple example, Q, K and V are all X.
    Q = X
    K = X
    V = X

    # Compare each token with every other token.
    scores = Q @ K.T / np.sqrt(X.shape[1])

    # Convert scores into attention weights.
    weights = softmax(scores)

    # Combine token information using the attention weights.
    output = weights @ V

    return output, weights


# ---------------- TASK 1 ----------------

# Create 6 tokens, each represented by 8 numbers.
X = np.random.rand(6, 8)

# Generate position information for all 6 tokens.
PE = positional_encoding(seq_len=6, d_model=8)

# Add position information to the original embeddings.
X_with_position = X + PE

# Calculate attention before adding position information.
_, weights_without_position = self_attention(X)

# Calculate attention after adding position information.
_, weights_with_position = self_attention(X_with_position)

# Print the two attention matrices side by side.
print("Attention WITHOUT positional encoding:")
print(np.round(weights_without_position, 3))

print("\nAttention WITH positional encoding:")
print(np.round(weights_with_position, 3))