import numpy as np

# Parameters for two layers
hidden_size_1 = 5  # Hidden units in layer 1
hidden_size_2 = 3  # Hidden units in layer 2
output_size = 1    # Output size

# Initialize weights and biases
W_x1 = np.random.randn(hidden_size_1, 1) * 0.01
W_h1 = np.random.randn(hidden_size_1, hidden_size_1) * 0.01
b_h1 = np.zeros((hidden_size_1, 1))

W_x2 = np.random.randn(hidden_size_2, hidden_size_1) * 0.01
W_h2 = np.random.randn(hidden_size_2, hidden_size_2) * 0.01
b_h2 = np.zeros((hidden_size_2, 1))

W_y = np.random.randn(output_size, hidden_size_2) * 0.01
b_y = np.zeros((output_size, 1))

# Activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Loss function
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

# Gradient of the loss w.r.t output
def mse_loss_grad(y_pred, y_true):
    return 2 * (y_pred - y_true) / len(y_true)

# Input sequence and target outputs
X = [0.5, 0.8, 0.2]  # Input sequence
y_true = [0.4, 0.6, 0.3]  # Target outputs

# Training parameters
learning_rate = 0.01
epochs = 100

# Training loop
for epoch in range(epochs):
    total_loss = 0
    h_prev_1 = np.zeros((hidden_size_1, 1))
    h_prev_2 = np.zeros((hidden_size_2, 1))
    
    # Store gradients
    grad_W_x1 = np.zeros_like(W_x1)
    grad_W_h1 = np.zeros_like(W_h1)
    grad_b_h1 = np.zeros_like(b_h1)

    grad_W_x2 = np.zeros_like(W_x2)
    grad_W_h2 = np.zeros_like(W_h2)
    grad_b_h2 = np.zeros_like(b_h2)

    grad_W_y = np.zeros_like(W_y)
    grad_b_y = np.zeros_like(b_y)

    for t, x_t in enumerate(X):
        x_t = np.array([[x_t]])  # Reshape input for matrix operations
        y_t_true = np.array([[y_true[t]]])  # True target
        
        # Forward pass
        h_t_1 = tanh(np.dot(W_x1, x_t) + np.dot(W_h1, h_prev_1) + b_h1)
        h_t_2 = tanh(np.dot(W_x2, h_t_1) + np.dot(W_h2, h_prev_2) + b_h2)
        y_t = np.dot(W_y, h_t_2) + b_y

        # Compute loss
        total_loss += mse_loss(y_t, y_t_true)
       
        # Backward pass
        dL_dy = mse_loss_grad(y_t, y_t_true)  # Gradient of loss w.r.t y_t
        grad_W_y += np.dot(dL_dy, h_t_2.T)  # Gradient for W_y
        grad_b_y += dL_dy  # Gradient for b_y
        
        # Gradients for second hidden layer
        dL_dh2 = np.dot(W_y.T, dL_dy) * tanh_derivative(h_t_2)
        grad_W_x2 += np.dot(dL_dh2, h_t_1.T)
        grad_W_h2 += np.dot(dL_dh2, h_prev_2.T)
        grad_b_h2 += dL_dh2

        # Gradients for first hidden layer
        dL_dh1 = np.dot(W_x2.T, dL_dh2) * tanh_derivative(h_t_1)
        grad_W_x1 += np.dot(dL_dh1, x_t.T)
        grad_W_h1 += np.dot(dL_dh1, h_prev_1.T)
        grad_b_h1 += dL_dh1

        # Update hidden states for next time step
        h_prev_1 = h_t_1
        h_prev_2 = h_t_2

    # Average loss
    total_loss /= len(X)

    # Update weights and biases
    W_x1 -= learning_rate * grad_W_x1
    W_h1 -= learning_rate * grad_W_h1
    b_h1 -= learning_rate * grad_b_h1

    W_x2 -= learning_rate * grad_W_x2
    W_h2 -= learning_rate * grad_W_h2
    b_h2 -= learning_rate * grad_b_h2

    W_y -= learning_rate * grad_W_y
    b_y -= learning_rate * grad_b_y

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.5f}")

epochs