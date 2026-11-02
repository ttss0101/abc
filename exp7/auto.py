import numpy as np

# Original pattern
X = np.array([1, -1, 1])

# Create weight matrix
W = np.outer(X, X)

# Remove self connections
np.fill_diagonal(W, 0)

print("Weight Matrix:")
print(W)

# Noisy input pattern
input_pattern = np.array([1, 1, 1])

# Recall
net = np.dot(input_pattern, W)

# Apply sign activation
output = np.where(net > 0, 1,
         np.where(net < 0, -1, input_pattern))

print("\nInput Pattern:")
print(input_pattern)

print("\nNet Output:")
print(net)

print("\nRecalled Pattern:")
print(output)