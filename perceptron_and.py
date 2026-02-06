import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# AND gate dataset (integers)
# x1, x2, bias(=1)
# -----------------------------
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([0, 0, 0, 1])  # AND gate outputs

# -----------------------------
# Initialize weights and params
# -----------------------------
weights = np.zeros(3)      # w1, w2, bias_weight
learning_rate = 1

# Step activation function
def step(net):
    return 1 if net >= 0 else 0

# -----------------------------
# Plot setup (4 quadrant graph)
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Perceptron Training for AND Gate")

def plot_decision_boundary(w):
    x_vals = np.array([-2, 2])
    if w[1] != 0:
        y_vals = -(w[0]*x_vals + w[2]) / w[1]
        ax.plot(x_vals, y_vals, 'g--')

# -----------------------------
# Training loop
# -----------------------------
epoch = 0
while True:
    error_count = 0
    epoch += 1
    print(f"\nEpoch {epoch}")

    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Perceptron Training for AND Gate")

    for i in range(len(X)):
        net = np.dot(weights, X[i])
        y_pred = step(net)
        error = y[i] - y_pred

        # Plot points: correct = blue, incorrect = red
        if error == 0:
            ax.scatter(X[i][0], X[i][1], color='blue', s=100, marker='o')
        else:
            ax.scatter(X[i][0], X[i][1], color='red', s=100, marker='x')
            error_count += 1

            # Weight update
            weights = weights + learning_rate * error * X[i]
            print(f"Updated weights after input {X[i][:2]}: {weights}")

    plot_decision_boundary(weights)
    plt.pause(0.7)

    # Stop if no errors in full pass
    if error_count == 0:
        break

# -----------------------------
# Final result
# -----------------------------
plt.ioff()
print("\nTraining converged!")
print("Final weights:", weights)
plt.show()
