import math
import pandas as pd


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Training dataset
data = [
    (0.8, 0.78, 1),
    (1.0, 0.67, 1),
    (0.9, 0.55, 1),
    (0.1, 0.20, 0),
    (1.2, 0.80, 1),
    (1.1, 0.78, 1),
    (0.4, 0.80, 0),
    (0.8, 0.75, 1),
    (0.1, 0.10, 0),
    (0.2, 0.40, 0)
]


# Initial weights
v01, v11, v21 = 0.3, 0.6, -0.1
v02, v12, v22 = 0.5, -0.3, 0.4

w0, w1, w2 = -0.2, 0.4, 0.1

# Learning rate
alpha = 0.25

results = []

for iteration, (x1, x2, t) in enumerate(data, 1):

    # Forward propagation

    z1in = v01 + x1 * v11 + x2 * v21
    z2in = v02 + x1 * v12 + x2 * v22

    z1 = sigmoid(z1in)
    z2 = sigmoid(z2in)

    yin = w0 + z1 * w1 + z2 * w2
    y = sigmoid(yin)

    # Output layer

    error = t - y
    delta = error * y * (1 - y)

    dw0 = alpha * delta
    dw1 = alpha * delta * z1
    dw2 = alpha * delta * z2

    # Hidden layer

    d1 = delta * w1 * z1 * (1 - z1)
    d2 = delta * w2 * z2 * (1 - z2)

    dv01 = alpha * d1
    dv11 = alpha * d1 * x1
    dv21 = alpha * d1 * x2

    dv02 = alpha * d2
    dv12 = alpha * d2 * x1
    dv22 = alpha * d2 * x2

    # Update weights

    w0 += dw0
    w1 += dw1
    w2 += dw2

    v01 += dv01
    v11 += dv11
    v21 += dv21

    v02 += dv02
    v12 += dv12
    v22 += dv22

    # Store results

    results.append([
        iteration, x1, x2, t,
        z1in, z2in, z1, z2, yin, y,
        error, delta,
        dw0, dw1, dw2,
        dv01, dv02, dv11, dv12, dv21, dv22,
        w0, w1, w2,
        v01, v02, v11, v12, v21, v22
    ])


# Display results

columns = [
    "It", "X1", "X2", "T",
    "Z1in", "Z2in", "Z1", "Z2",
    "Yin", "Y", "T-Y", "Delta",
    "dW0", "dW1", "dW2",
    "dV01", "dV02", "dV11", "dV12",
    "dV21", "dV22",
    "W0", "W1", "W2",
    "V01", "V02", "V11", "V12",
    "V21", "V22"
]

df = pd.DataFrame(results, columns=columns)

print("\nBack Propagation Results\n")
print(df.round(4).to_string(index=False))