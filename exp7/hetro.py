import numpy as np

# Input patterns
X = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])

# Corresponding output patterns
Y = np.array([
    [1, -1],
    [-1, 1]
])

# Calculate weight matrix
W = np.dot(X.T, Y)

print("Weight Matrix:")
print(W)

# Recall output for first input
input_pattern = X[0]

# Calculate output
output = np.dot(input_pattern, W)

# Apply bipolar activation function
output = np.where(output >= 0, 1, -1)

print("\nInput Pattern:")
print(input_pattern)

print("\nRecalled Output:")
print(output)