import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define inputs and output
water = ctrl.Antecedent(np.arange(0, 101, 1), 'water')
rate = ctrl.Antecedent(np.arange(-10, 11, 1), 'rate')
gate = ctrl.Consequent(np.arange(0, 101, 1), 'gate')

# Membership functions for water level
water['Low'] = fuzz.trimf(water.universe, [0, 0, 50])
water['Medium'] = fuzz.trimf(water.universe, [25, 50, 75])
water['High'] = fuzz.trimf(water.universe, [50, 100, 100])

# Membership functions for rate
rate['Falling'] = fuzz.trimf(rate.universe, [-10, -10, 0])
rate['Stable'] = fuzz.trimf(rate.universe, [-5, 0, 5])
rate['Rising'] = fuzz.trimf(rate.universe, [0, 10, 10])

# Membership functions for gate
gate['Closed'] = fuzz.trimf(gate.universe, [0, 0, 40])
gate['Partial'] = fuzz.trimf(gate.universe, [20, 50, 80])
gate['Open'] = fuzz.trimf(gate.universe, [60, 100, 100])

# Define rules
rule1 = ctrl.Rule(water['Low'] & rate['Falling'], gate['Closed'])
rule2 = ctrl.Rule(water['Low'] & rate['Stable'], gate['Closed'])
rule3 = ctrl.Rule(water['Low'] & rate['Rising'], gate['Partial'])

rule4 = ctrl.Rule(water['Medium'] & rate['Falling'], gate['Partial'])
rule5 = ctrl.Rule(water['Medium'] & rate['Stable'], gate['Partial'])
rule6 = ctrl.Rule(water['Medium'] & rate['Rising'], gate['Open'])

rule7 = ctrl.Rule(water['High'] & rate['Falling'], gate['Partial'])
rule8 = ctrl.Rule(water['High'] & rate['Stable'], gate['Open'])
rule9 = ctrl.Rule(water['High'] & rate['Rising'], gate['Open'])

# Create control system
system = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])

simulation = ctrl.ControlSystemSimulation(system)

# Take user input
w = float(input("Enter water level (0-100): "))
r = float(input("Enter rate of change (-10 to 10): "))

# Pass inputs
simulation.input['water'] = w
simulation.input['rate'] = r

# Calculate output
simulation.compute()

# Display result
print("Gate Opening:", simulation.output['gate'], "%")