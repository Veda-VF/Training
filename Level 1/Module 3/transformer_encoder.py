"""
A Transformer encoder layer takes token representations, lets them share information through attention, processes them through a feed-forward network, and uses residual connections and layer normalization to help keep the computation stable.
Component	             What it does
---------------------   ------------------------------------------------------
Self-attention	         Lets each token use information from other tokens
Residual connection	     Adds the original input back to the processed output
Layer normalization	     Normalizes each token's feature values
Feed-forward network	 Further processes each token's features
L2 norm	                 Measures the overall size of the output values


flowchart
    A([Start]) --> B[Create input]
    B --> C[Self-attention]
    B --> D[Keep original input for skip connection]
    C --> E[Add original input]
    D --> E
    E --> F[Layer normalization]
    F --> G[Feed-forward network]
    F --> H[Keep normalized input for skip connection]
    G --> I[Add normalized input]
    H --> I
    I --> J[Layer normalization]
    J --> K[Print output shape and L2 norm]
    K --> L{Three layers completed?}
    L -- No --> C
    L -- Yes --> [End]

"""
import numpy as np

np.random.seed(42)

# Convert scores into attention weights.
def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


# Normalize the feature values of each token.
def layer_norm(X):
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)

    return (X - mean) / np.sqrt(variance + 1e-6)


# Build one complete Transformer encoder layer.
def transformer_encoder_layer(X):
    d_model = X.shape[1]

    # STEP 1: Self-attention
    # Each token compares its features with every token.
    Q = X
    K = X
    V = X

    scores = Q @ K.T / np.sqrt(d_model)
    attention_weights = softmax(scores)
    attention_output = attention_weights @ V

    # STEP 2: Residual connection + normalization
    # Add the original input to the attention output.
    X = layer_norm(X + attention_output)

    # STEP 3: Feed-forward network
    # Expand 16 features to 32, then return to 16.
    W1 = np.random.randn(d_model, 32) * 0.1
    b1 = np.zeros(32)

    W2 = np.random.randn(32, d_model) * 0.1
    b2 = np.zeros(d_model)

    hidden = np.maximum(0, X @ W1 + b1)  # ReLU activation
    ff_output = hidden @ W2 + b2

    # STEP 4: Residual connection + normalization
    X = layer_norm(X + ff_output)

    return X


# ---------------- TASK 2 ----------------

# Create 4 tokens, each having 16 features.
X = np.random.randn(4, 16)

print("Initial shape:", X.shape)
print("Initial L2 norm:", round(np.linalg.norm(X), 3))

# Pass the output through 3 encoder layers.
for layer_number in range(1, 4):
    X = transformer_encoder_layer(X)

    print(f"\nAfter encoder layer {layer_number}:")
    print("Output shape:", X.shape)
    print("L2 norm:", round(np.linalg.norm(X), 3))

print("\nAll 3 encoder layers completed!")