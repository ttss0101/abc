import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# 1. Generate sample traffic data
# --------------------------------

np.random.seed(42)

n = 200

vehicles = np.random.uniform(0, 200, n)
speed = np.random.uniform(5, 100, n)

# Simplified traffic flow score
traffic = (
    0.6 * (vehicles / 200) +
    0.4 * (1 - speed / 100)
) * 100

traffic = np.clip(traffic, 0, 100)

X = np.column_stack((vehicles, speed))
y = traffic

# --------------------------------
# 2. Normalize input data
# --------------------------------

minimum = X.min(axis=0)
maximum = X.max(axis=0)

X = (X - minimum) / (maximum - minimum)

# --------------------------------
# 3. Split the dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 4. Gaussian membership function
# --------------------------------

def gaussian(x, center, sigma):
    return np.exp(
        -((x - center) ** 2) / (2 * sigma ** 2)
    )

# --------------------------------
# 5. Calculate rule firing strengths
# --------------------------------

def firing_strengths(X):

    centers = [0.25, 0.75]
    sigma = 0.30

    rules = []

    for c1 in centers:
        for c2 in centers:

            m1 = gaussian(X[:, 0], c1, sigma)
            m2 = gaussian(X[:, 1], c2, sigma)

            rules.append(m1 * m2)

    return np.array(rules).T

# --------------------------------
# 6. Normalize firing strengths
# --------------------------------

def normalize(weights):

    return weights / (
        np.sum(weights, axis=1, keepdims=True) + 1e-10
    )

# --------------------------------
# 7. Create ANFIS training matrix
# --------------------------------

def create_matrix(X):

    weights = normalize(firing_strengths(X))

    matrix = []

    for i in range(4):

        w = weights[:, i]

        row = np.column_stack((
            w * X[:, 0],
            w * X[:, 1],
            w
        ))

        matrix.append(row)

    return np.hstack(matrix)

# --------------------------------
# 8. Train the model
# --------------------------------

A = create_matrix(X_train)

parameters = np.linalg.lstsq(
    A, y_train, rcond=None
)[0]

# --------------------------------
# 9. Predict traffic flow
# --------------------------------

def predict(X):

    A = create_matrix(X)

    return A @ parameters

# --------------------------------
# 10. Test the model
# --------------------------------

predictions = predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("ANFIS TRAFFIC PREDICTION")
print("------------------------")
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# --------------------------------
# 11. Take user input
# --------------------------------

print("\nEnter Traffic Details")

v = float(input("Enter number of vehicles: "))
s = float(input("Enter average speed (km/h): "))

user_data = np.array([[v, s]])

user_data = (
    user_data - minimum
) / (maximum - minimum)

result = predict(user_data)[0]

result = np.clip(result, 0, 100)

print("\nPredicted Traffic Score:", round(result, 2))

if result < 35:
    print("Traffic Level: LOW")
elif result < 65:
    print("Traffic Level: MEDIUM")
else:
    print("Traffic Level: HIGH")