import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(x):
    return x**2

def gradient(x):
    return 2 * x

# Gradient Descent Parameters
x_start = 10  # Starting point
learning_rate = 0.1
epochs = 20

# Track progress
x_values = [x_start]
f_values = [f(x_start)]

# Gradient Descent Loop
x = x_start
for i in range(epochs):
    grad = gradient(x)
    x = x - learning_rate * grad  # Update rule
    x_values.append(x)
    f_values.append(f(x))

# Plotting the Function and Descent Path
x_range = np.linspace(-10, 10, 100)
plt.figure(figsize=(10, 6))
plt.plot(x_range, f(x_range), label="f(x) = x^2")
plt.scatter(x_values, f_values, color="red", label="Gradient Descent Steps")
plt.plot(x_values, f_values, linestyle="--", color="red")
plt.title("Gradient Descent on f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

# Final Results
print(f"Final x value: {x}")
print(f"Final f(x) value: {f(x)}")
