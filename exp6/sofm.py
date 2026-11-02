import numpy as np

# Input data
inputs = np.array([
    [2, 3],
    [4, 5],
    [1, 2]
])

# Initial weights of neurons
weights = np.array([
    [1, 2],
    [4, 4]
], dtype=float)

# Learning rate
alpha = 0.5

# Training
for x in inputs:

    print("\nInput:", x)

    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((weights - x) ** 2, axis=1))

    print("Distances:", distances)

    # Find winning neuron
    winner = np.argmin(distances)

    print("Winning Neuron:", winner + 1)

    # Update winning neuron's weights
    weights[winner] = weights[winner] + alpha * (x - weights[winner])

    print("Updated Weights:")
    print(weights)

print("\nFinal Weights:")
print(weights)