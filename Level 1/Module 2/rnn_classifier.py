
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Keep results more repeatable
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 1. Dataset settings
SEQUENCE_LENGTH = 20
TRAIN_SIZE = 500
TEST_SIZE = 100


# 2. Check whether a sequence contains "111"
def contains_111(sequence):
    for i in range(len(sequence) - 2):
        if sequence[i:i + 3] == [1, 1, 1]:
            return True
    return False


# 3. Generate synthetic sequences and their labels
def create_dataset(size):
    sequences = []
    labels = []

    for _ in range(size):
        # Generate a random sequence of 20 zeros and ones
        sequence = np.random.randint(
            0, 2, size=SEQUENCE_LENGTH
        ).tolist()

        # 1 if "111" occurs; otherwise 0
        label = int(contains_111(sequence))

        sequences.append(sequence)
        labels.append(label)

    # RNN input shape: (samples, time steps, features)
    X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


X_train, y_train = create_dataset(TRAIN_SIZE)
X_test, y_test = create_dataset(TEST_SIZE)

# Process training data in small batches
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True
)


# 4. Build the RNN classifier
class PatternRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Read the sequence one value at a time
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,
            batch_first=True
        )

        # Predict one of two classes: 0 or 1
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        # hidden stores the final hidden state
        _, hidden = self.rnn(x)

        # Remove the extra first dimension
        hidden = hidden.squeeze(0)

        # Return scores for both classes
        return self.fc(hidden)


model = PatternRNN()

# Loss measures how far predictions are from correct labels
criterion = nn.CrossEntropyLoss()

# Adam updates the model's learnable parameters
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)


# 5. Train the model
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for sequences, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass: predict classes
        outputs = model(sequences)

        # Calculate prediction error
        loss = criterion(outputs, labels)

        # Backpropagation: calculate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item() * len(labels)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {total_loss / TRAIN_SIZE:.4f}"
        )


# 6. Evaluate on the 100 unseen test sequences
model.eval()

with torch.no_grad():
    outputs = model(X_test)

    # Select the class with the highest score
    predictions = torch.argmax(outputs, dim=1)

correct = (predictions == y_test).sum().item()
accuracy = correct / TEST_SIZE * 100

print(f"\nTest Accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {correct}/{TEST_SIZE}")


# 7. Display a few sample predictions
print("\nSample predictions:")

for i in range(5):
    sequence = X_test[i, :, 0].to(torch.int64).tolist()

    print(
        f"Sequence: {''.join(map(str, sequence))} | "
        f"Actual: {y_test[i].item()} | "
        f"Predicted: {predictions[i].item()}"
    )
