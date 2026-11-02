import numpy as np

# =====================================
# EXPERIMENT 8
# FUZZY SETS AND FUZZY RELATIONS
# =====================================

# Define two fuzzy sets
A = np.array([0.2, 0.4, 0.6, 0.8])
B = np.array([0.5, 0.3, 0.7, 0.9])

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)


# =====================================
# 1. FUZZY SET OPERATIONS
# =====================================

# Union
union = np.maximum(A, B)

# Intersection
intersection = np.minimum(A, B)

# Complement
complement = 1 - A

# Algebraic Sum
algebraic_sum = A + B - (A * B)

# Algebraic Product
algebraic_product = A * B

# Bounded Sum
bounded_sum = np.minimum(1, A + B)

# Bounded Difference
bounded_difference = np.maximum(0, A - B)


print("\n--- FUZZY SET OPERATIONS ---")

print("Union:", union)
print("Intersection:", intersection)
print("Complement:", complement)
print("Algebraic Sum:", algebraic_sum)
print("Algebraic Product:", algebraic_product)
print("Bounded Sum:", bounded_sum)
print("Bounded Difference:", bounded_difference)


# =====================================
# 2. CARTESIAN PRODUCT
# =====================================

R = np.minimum(
    A.reshape(-1, 1),
    B.reshape(1, -1)
)

print("\n--- CARTESIAN PRODUCT ---")
print(R)


# =====================================
# 3. MAX-MIN COMPOSITION
# =====================================

R1 = np.array([
    [0.2, 0.5, 0.7],
    [0.4, 0.8, 0.3]
])

R2 = np.array([
    [0.6, 0.3],
    [0.7, 0.9],
    [0.5, 0.4]
])
result = np.zeros((R1.shape[0], R2.shape[1]))
print(result)


def max_min(R1, R2):

    result = np.zeros((R1.shape[0], R2.shape[1]))

    for i in range(R1.shape[0]):
        for j in range(R2.shape[1]):

            values = np.minimum(R1[i, :], R2[:, j])

            result[i, j] = np.max(values)

    return result


print("\n--- MAX-MIN COMPOSITION ---")
print(max_min(R1, R2))


# =====================================
# 4. MAX-PRODUCT COMPOSITION
# =====================================

def max_product(R1, R2):

    result = np.zeros((R1.shape[0], R2.shape[1]))

    for i in range(R1.shape[0]):
        for j in range(R2.shape[1]):

            values = R1[i, :] * R2[:, j]

            result[i, j] = np.max(values)

    return result


print("\n--- MAX-PRODUCT COMPOSITION ---")
print(max_product(R1, R2))