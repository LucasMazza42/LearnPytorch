import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Data
def generate_sine_wave(seq_length=50, num_samples=1000):
    x = np.linspace(0, 50 * np.pi, num_samples)
    y = np.sin(x)
    X, Y = [], []
    for i in range(len(y) - seq_length):
        X.append(y[i:i+seq_length])  # Input sequence
        Y.append(y[i+seq_length])   # Next value (target)
    return np.array(X), np.array(Y)


# Parameters
seq_length = 50
data = generate_sine_wave(seq_length=seq_length)

# Split into training and testing data
split = int(len(data) * 0.8)
train_data, test_data = data[:split], data[split:]
X_train = torch.tensor([d[0] for d in train_data], dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor([d[1] for d in train_data], dtype=torch.float32)
X_test = torch.tensor([d[0] for d in test_data], dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor([d[1] for d in test_data], dtype=torch.float32)

# Step 2: Define the RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # RNN layer
        if len(out.shape) == 2:  # Handle single sequence input
            out = out[-1, :]  # Last hidden state
        else:
            out = out[:, -1, :]  # Batch case: last hidden state of each sequence
        out = self.fc(out)  # Fully connected layer
        return out


# Step 3: Initialize RNN, Loss, and Optimizer
input_size = 1
hidden_size = 10
output_size = 1

rnn = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Step 4: Train the RNN
epochs = 50
losses = []

for epoch in range(epochs):
    rnn.train()
    optimizer.zero_grad()
    outputs = rnn(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.5f}")

# Step 5: Evaluate and Visualize
rnn.eval()
with torch.no_grad():
    predictions = rnn(X_test).squeeze()

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(losses, label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test.numpy(), label="Actual")
plt.plot(predictions.numpy(), label="Predicted")
plt.title("RNN Predictions vs Actual")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
