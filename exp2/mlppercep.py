# -------------------------------
# MLP for XOR Problem using NumPy
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ XOR Dataset
# -------------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# -------------------------------
# 2️⃣ Activation Function
# -------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -------------------------------
# 3️⃣ Initialize Network Parameters
# -------------------------------
np.random.seed(42)

input_size = 2
hidden_size = 4
output_size = 1

# Weights
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# -------------------------------
# 4️⃣ Hyperparameters
# -------------------------------
learning_rate = 0.1
epochs = 10000
losses = []

# -------------------------------
# 5️⃣ Training MLP (Backpropagation)
# -------------------------------
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(final_input)
    
    # Compute Loss (MSE)
    loss = np.mean((y - output)**2)
    losses.append(loss)
    
    # Backpropagation
    d_output = (y - output) * sigmoid_derivative(output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    
    # Update Weights and Biases
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# -------------------------------
# 6️⃣ Final Predictions
# -------------------------------
print("\nFinal Predictions (after training):")
print(output)

# -------------------------------
# 7️⃣ Plot Loss Curve
# -------------------------------
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.show()

# -------------------------------
# 8️⃣ Plot Decision Boundary
# -------------------------------
def plot_decision_boundary():
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    hidden = sigmoid(np.dot(grid, W1) + b1)
    out = sigmoid(np.dot(hidden, W2) + b2)
    Z = out.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['#FFAAAA','#AAAAFF'])
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), edgecolors='k', s=100)
    plt.title("Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

plot_decision_boundary()
