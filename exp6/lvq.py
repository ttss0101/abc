import numpy as np

# Training inputs
X = np.array([
    [2, 3],
    [4, 5],
    [1, 2],
    [5, 6]
])

# Target classes
Y = np.array([0, 1, 0, 1])

# Initial weights
weights = np.array([
    [1, 2],
    [4, 4]
], dtype=float)

# Class of each neuron
classes = np.array([0, 1])

# Learning rate
alpha = 0.5

# Training
for i in range(len(X)):

    x = X[i]
    target = Y[i]

    print("\nInput:", x)
    print("Target Class:", target)

    # Calculate distances
    distances = np.sqrt(np.sum((weights - x) ** 2, axis=1))

    print("Distances:", distances)

    # Find winning neuron
    winner = np.argmin(distances)

    print("Winning Neuron:", winner + 1)

    # Check whether winner has correct class
    if classes[winner] == target:

        # Move weights towards input
        weights[winner] = weights[winner] + alpha * (x - weights[winner])

        print("Correct Winner: Moving weights closer")

    else:

        # Move weights away from input
        weights[winner] = weights[winner] - alpha * (x - weights[winner])

        print("Incorrect Winner: Moving weights away")

    print("Updated Weights:")
    print(weights)

print("\nFinal Weights:")
print(weights)