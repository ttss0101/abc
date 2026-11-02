import numpy as np

# Input pattern
X = np.array([1, -1])

# Associated output pattern
Y = np.array([1, 1])

# Calculate weight matrix
W = np.outer(X, Y)

print("Weight Matrix:")
print(W)

# Forward recall: X to Y
net_y = np.dot(X, W)

output_y = np.where(net_y >= 0, 1, -1)

print("\nForward Recall (X to Y):")
print(output_y)

# Backward recall: Y to X
net_x = np.dot(output_y, W.T)

output_x = np.where(net_x >= 0, 1, -1)

print("\nBackward Recall (Y to X):")
print(output_x)