import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define inputs and output
error = ctrl.Antecedent(np.arange(-50, 51, 1), 'error')
acceleration = ctrl.Antecedent(np.arange(-10, 11, 1), 'acceleration')
power = ctrl.Consequent(np.arange(0, 101, 1), 'power')

# Membership functions for speed error
error['Negative'] = fuzz.trimf(error.universe, [-50, -50, 0])
error['Zero'] = fuzz.trimf(error.universe, [-20, 0, 20])
error['Positive'] = fuzz.trimf(error.universe, [0, 50, 50])

# Membership functions for acceleration
acceleration['Decelerating'] = fuzz.trimf(acceleration.universe, [-10, -10, 0])
acceleration['Stable'] = fuzz.trimf(acceleration.universe, [-5, 0, 5])
acceleration['Accelerating'] = fuzz.trimf(acceleration.universe, [0, 10, 10])

# Membership functions for motor power
power['Low'] = fuzz.trimf(power.universe, [0, 0, 40])
power['Medium'] = fuzz.trimf(power.universe, [20, 50, 80])
power['High'] = fuzz.trimf(power.universe, [60, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(error['Negative'] & acceleration['Accelerating'], power['Low'])
rule2 = ctrl.Rule(error['Negative'] & acceleration['Stable'], power['Low'])
rule3 = ctrl.Rule(error['Negative'] & acceleration['Decelerating'], power['Low'])

rule4 = ctrl.Rule(error['Zero'] & acceleration['Stable'], power['Medium'])

rule5 = ctrl.Rule(error['Positive'] & acceleration['Decelerating'], power['High'])
rule6 = ctrl.Rule(error['Positive'] & acceleration['Stable'], power['High'])
rule7 = ctrl.Rule(error['Positive'] & acceleration['Accelerating'], power['Medium'])

# Create control system
system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4,
    rule5, rule6, rule7
])

simulation = ctrl.ControlSystemSimulation(system)

# Take user input
e = float(input("Enter speed error: "))
a = float(input("Enter acceleration: "))

# Pass inputs
simulation.input['error'] = e
simulation.input['acceleration'] = a

# Calculate output
simulation.compute()

# Display result
print("Motor Power:", simulation.output['power'], "%")