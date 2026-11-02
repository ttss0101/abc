import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# 1. CREATE STOCK DATASET
# --------------------------------

data = {
    'Price': [100, 110, 120, 130, 140, 150, 160, 170,
              180, 190, 200, 210, 220, 230, 240, 250,
              260, 270, 280, 290],

    'Volume': [1000, 1200, 1500, 1300, 1800, 2000, 2200,
               2100, 2500, 2700, 3000, 2800, 3200, 3500,
               3300, 3800, 4000, 4200, 4500, 4700],

    'NextPrice': [105, 115, 125, 135, 145, 155, 165, 175,
                  185, 195, 205, 215, 225, 235, 245, 255,
                  265, 275, 285, 295]
}

df = pd.DataFrame(data)

# Inputs and output
X = df[['Price', 'Volume']].values
y = df['NextPrice'].values

# --------------------------------
# 2. NORMALIZE INPUT DATA
# --------------------------------

minimum = X.min(axis=0)
maximum = X.max(axis=0)

X = (X - minimum) / (maximum - minimum)

# --------------------------------
# 3. SPLIT DATA
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 4. GAUSSIAN MEMBERSHIP FUNCTION
# --------------------------------

def membership(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


# --------------------------------
# 5. DEFINE MEMBERSHIP FUNCTIONS
# --------------------------------

centers = [0.25, 0.75]
sigma = 0.35

# Generate 4 fuzzy rules
rules = []

for i in range(2):
    for j in range(2):
        rules.append((i, j))


# --------------------------------
# 6. CALCULATE FIRING STRENGTHS
# --------------------------------

def calculate_weights(row):

    weights = []

    for i, j in rules:

        m1 = membership(row[0], centers[i], sigma)
        m2 = membership(row[1], centers[j], sigma)

        weights.append(m1 * m2)

    weights = np.array(weights)

    return weights / (np.sum(weights) + 1e-10)


# --------------------------------
# 7. TRAIN ANFIS MODEL
# --------------------------------

def create_matrix(X):

    matrix = []

    for row in X:

        weights = calculate_weights(row)

        features = []

        for w in weights:

            features.extend([
                w * row[0],
                w * row[1],
                w
            ])

        matrix.append(features)

    return np.array(matrix)


A = create_matrix(X_train)

# Least squares training of consequent parameters
parameters = np.linalg.lstsq(A, y_train, rcond=None)[0]

print("ANFIS Model Training Completed!")


# --------------------------------
# 8. PREDICTION FUNCTION
# --------------------------------

def predict(X):

    A = create_matrix(X)

    return A @ parameters


# --------------------------------
# 9. TEST THE MODEL
# --------------------------------

y_pred = predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTesting Results")
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 4))


# --------------------------------
# 10. PREDICT NEW STOCK PRICE
# --------------------------------

price = float(input("\nEnter previous closing price: "))
volume = float(input("Enter trading volume: "))

new_data = np.array([[price, volume]])

new_data = (new_data - minimum) / (maximum - minimum)

result = predict(new_data)

print("\nPredicted Next Closing Price: Rs.", round(result[0], 2))