import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Step activation function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# Perceptron class
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def train(self, X, y, title):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        errors = []

        plt.figure(figsize=(6, 5))

        for epoch in range(self.epochs):
            total_error = 0
            print(f"\nEpoch {epoch + 1}")
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = step_function(linear_output)
                error = y[i] - y_pred

                if error != 0:
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error

                print(f"Input:{X[i]} Target:{y[i]} "
                      f"Pred:{y_pred} Weights:{self.weights} Bias:{self.bias}")

                total_error += abs(error)

            errors.append(total_error)
            self.plot_decision_boundary(X, y, title, epoch)

            if total_error == 0:
                print("Model Converged!")
                break
            else:
              print ("model cannot converge for XOR dataset!")

        return errors

    def plot_decision_boundary(self, X, y, title, epoch):
        plt.clf()
        for label, marker, color in zip([0, 1], ['o', 's'], ['red', 'blue']):
            plt.scatter(
                X[y == label][:, 0],
                X[y == label][:, 1],
                marker=marker,
                color=color,
                label=f"Class {label}"
            )

        x_vals = np.array([-0.2, 1.2])
        if self.weights[1] != 0:
            y_vals = -(self.weights[0] * x_vals + self.bias) / self.weights[1]
            plt.plot(x_vals, y_vals, 'k--')

        plt.title(f"{title} - Epoch {epoch + 1}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.pause(0.5)

# NAND Dataset
X_nand = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_nand = np.array([1, 1, 1, 0])

# XOR Dataset
X_xor = X_nand.copy()
y_xor = np.array([0, 1, 1, 0])

# Train NAND
print("\n=== Training NAND Gate ===")
perceptron_nand = Perceptron(lr=0.1, epochs=20)
errors_nand = perceptron_nand.train(X_nand, y_nand, "NAND Perceptron")

plt.figure()
plt.plot(errors_nand, marker='o')
plt.title("Training Error Curve - NAND")
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.show()

# Train XOR
print("\n=== Training XOR Gate ===")
perceptron_xor = Perceptron(lr=0.1, epochs=10)
errors_xor = perceptron_xor.train(X_xor, y_xor, "XOR Perceptron")

plt.figure()
plt.plot(errors_xor, marker='o', color='red')
plt.title("Training Error Curve - XOR")
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.show()
