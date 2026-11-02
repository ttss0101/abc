import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# 1. CREATE AQI DATASET
# --------------------------------

data = {
    'PM25': [20, 30, 40, 50, 60, 70, 80, 90,
             100, 110, 120, 130, 140, 150, 160,
             170, 180, 190, 200, 210],

    'PM10': [30, 40, 50, 60, 70, 80, 90, 100,
             110, 120, 130, 140, 150, 160, 170,
             180, 190, 200, 210, 220],

    'AQI': [35, 45, 55, 65, 75, 85, 95, 105,
            115, 125, 135, 145, 155, 165, 175,
            185, 195, 205, 215, 225]
}

df = pd.DataFrame(data)

# Inputs and output
X = df[['PM25', 'PM10']].values
y = df['AQI'].values

# --------------------------------
# 2. SPLIT DATA
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# 3. DEFINE MEMBERSHIP FUNCTION
# --------------------------------

def membership(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


# --------------------------------
# 4. INITIALIZE ANFIS PARAMETERS
# --------------------------------

centers = np.array([
    [30, 50, 70],
    [40, 100, 160]
], dtype=float)

sigma = np.array([30, 60], dtype=float)

# --------------------------------
# 5. GENERATE FUZZY RULES
# --------------------------------

rules = []

for i in range(3):
    for j in range(3):
        rules.append((i, j))

# --------------------------------
# 6. CALCULATE ANFIS OUTPUT
# --------------------------------

def predict(X):

    outputs = []

    for row in X:

        pm25 = row[0]
        pm10 = row[1]

        firing = []

        for i, j in rules:

            m1 = membership(pm25, centers[0][i], sigma[0])
            m2 = membership(pm10, centers[1][j], sigma[1])

            w = m1 * m2
            firing.append(w)

        firing = np.array(firing)

        normalized = firing / (np.sum(firing) + 1e-10)

        rule_outputs = []

        for i, j in rules:

            rule_outputs.append(
                pm25 * (0.5 + 0.1 * i)
                + pm10 * (0.3 + 0.1 * j)
                + 5
            )

        final_output = np.sum(normalized * rule_outputs)

        outputs.append(final_output)

    return np.array(outputs)


# --------------------------------
# 7. TRAINING
# --------------------------------

print("Training ANFIS Model...")

for epoch in range(10):

    predicted = predict(X_train)

    error = np.mean((y_train - predicted) ** 2)

    print("Epoch", epoch + 1, "Error:", error)

print("Training Completed!")


# --------------------------------
# 8. TESTING
# --------------------------------

y_pred = predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTesting Results")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# --------------------------------
# 9. AQI PREDICTION
# --------------------------------

pm25 = float(input("\nEnter PM2.5 value: "))
pm10 = float(input("Enter PM10 value: "))

new_input = np.array([[pm25, pm10]])

predicted_aqi = predict(new_input)

print("\nPredicted AQI:", round(predicted_aqi[0], 2))