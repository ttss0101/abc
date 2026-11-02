import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# --------------------------------
# 1. Generate sample dataset
# --------------------------------

np.random.seed(42)

n = 200

glucose = np.random.uniform(70, 200, n)
bmi = np.random.uniform(18, 45, n)

# Generate simplified sample labels
risk = (
    0.65 * (glucose > 125) +
    0.25 * (bmi > 30) +
    np.random.uniform(0, 0.15, n)
)

diabetes = (risk >= 0.65).astype(int)

X = np.column_stack((glucose, bmi))
y = diabetes.astype(float)

# --------------------------------
# 2. Normalize inputs
# --------------------------------

minimum = X.min(axis=0)
maximum = X.max(axis=0)

X = (X - minimum) / (maximum - minimum)

# --------------------------------
# 3. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 4. Gaussian membership function
# --------------------------------

def gaussian(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


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
    return weights / (np.sum(weights, axis=1, keepdims=True) + 1e-10)


# --------------------------------
# 7. Train ANFIS model
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


A = create_matrix(X_train)

# Least squares training
parameters = np.linalg.lstsq(A, y_train, rcond=None)[0]


# --------------------------------
# 8. Prediction function
# --------------------------------

def predict(X):

    A = create_matrix(X)

    output = A @ parameters

    return np.clip(output, 0, 1)


# --------------------------------
# 9. Test the model
# --------------------------------

predictions = predict(X_test)

predicted_class = (predictions >= 0.5).astype(int)

accuracy = accuracy_score(
    y_test, predicted_class
)

mse = mean_squared_error(
    y_test, predictions
)

print("ANFIS MODEL RESULTS")
print("-------------------")
print("Accuracy:", accuracy * 100, "%")
print("Mean Squared Error:", mse)


# --------------------------------
# 10. Take user input
# --------------------------------

print("\nEnter Patient Details")

g = float(input("Enter Glucose Level: "))
b = float(input("Enter BMI: "))

user_data = np.array([[g, b]])

# Normalize user input using dataset values
user_data = (user_data - minimum) / (maximum - minimum)

result = predict(user_data)[0]

print("\nPredicted Risk Score:", result)

if result >= 0.5:
    print("Prediction: Diabetes Predicted")
else:
    print("Prediction: No Diabetes Predicted")