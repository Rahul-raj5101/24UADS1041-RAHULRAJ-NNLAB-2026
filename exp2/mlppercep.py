import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. XOR Dataset
# ----------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ----------------------------
# 2. Initialize Network
# ----------------------------
np.random.seed(1)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

# ----------------------------
# 3. Activation Functions
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ----------------------------
# 4. Training
# ----------------------------
learning_rate = 0.5
epochs = 4000
loss_list = []

for epoch in range(epochs):

    # Forward pass
    hidden_input = X @ W1 + b1
    hidden_output = sigmoid(hidden_input)

    final_input = hidden_output @ W2 + b2
    final_output = sigmoid(final_input)

    # Loss (Binary Cross Entropy-like MSE)
    loss = np.mean((y - final_output)**2)
    loss_list.append(loss)

    # Backpropagation
    error_output = (final_output - y) * sigmoid_derivative(final_output)
    error_hidden = error_output @ W2.T * sigmoid_derivative(hidden_output)

    # Update weights
    W2 -= learning_rate * hidden_output.T @ error_output
    b2 -= learning_rate * np.sum(error_output, axis=0, keepdims=True)

    W1 -= learning_rate * X.T @ error_hidden
    b1 -= learning_rate * np.sum(error_hidden, axis=0, keepdims=True)

    # ----------------------------
    # Print some epochs
    # ----------------------------
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

# ----------------------------
# 5. Final Predictions
# ----------------------------
print("\nFinal Predictions:")
predictions = (final_output > 0.5).astype(int)
print(predictions)

print("\nActual Output:")
print(y)

# ----------------------------
# 6. Plot Loss Curve
# ----------------------------
plt.plot(loss_list)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
