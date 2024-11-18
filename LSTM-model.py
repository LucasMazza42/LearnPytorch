import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Sequence
sequence = np.array([0.5, 0.1, -0.3])

# Initialize weights and biases (shared across time steps)
W_f, W_i, W_c, W_o = 0.5, 0.5, 0.5, 0.5  # Input weights
U_f, U_i, U_c, U_o = 0.3, 0.3, 0.3, 0.3  # Recurrent weights
b_f, b_i, b_c, b_o = 0.1, 0.1, 0.1, 0.1  # Biases

# Initial cell state and hidden state
c_prev = 0
h_prev = 0


for t, x_t in enumerate(sequence):
    # Forget gate
    f_t = sigmoid(W_f * x_t + U_f * h_prev + b_f)
    
    # Input gate
    i_t = sigmoid(W_i * x_t + U_i * h_prev + b_i)
    c_tilde_t = tanh(W_c * x_t + U_c * h_prev + b_c)
    
    # Cell state update
    c_t = f_t * c_prev + i_t * c_tilde_t
    
    # Output gate
    o_t = sigmoid(W_o * x_t + U_o * h_prev + b_o)
    
    # Hidden state update
    h_t = o_t * tanh(c_t)
    
    # Print results for this time step
    print(f"Time step {t + 1}")
    print(f"  Forget gate: {f_t:.3f}")
    print(f"  Input gate: {i_t:.3f}")
    print(f"  Cell candidate: {c_tilde_t:.3f}")
    print(f"  Cell state: {c_t:.3f}")
    print(f"  Output gate: {o_t:.3f}")
    print(f"  Hidden state: {h_t:.3f}\n")
    
    # Update previous states for the next time step
    c_prev = c_t
    h_prev = h_t
