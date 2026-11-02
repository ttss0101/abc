import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define inputs and output
tilt = ctrl.Antecedent(np.arange(-45, 46, 1), 'tilt')
velocity = ctrl.Antecedent(np.arange(-10, 11, 1), 'velocity')
correction = ctrl.Consequent(np.arange(-100, 101, 1), 'correction')

# Membership functions for tilt
tilt['Left'] = fuzz.trimf(tilt.universe, [-45, -45, 0])
tilt['Center'] = fuzz.trimf(tilt.universe, [-20, 0, 20])
tilt['Right'] = fuzz.trimf(tilt.universe, [0, 45, 45])

# Membership functions for velocity
velocity['Negative'] = fuzz.trimf(velocity.universe, [-10, -10, 0])
velocity['Zero'] = fuzz.trimf(velocity.universe, [-5, 0, 5])
velocity['Positive'] = fuzz.trimf(velocity.universe, [0, 10, 10])

# Membership functions for correction
correction['StrongLeft'] = fuzz.trimf(correction.universe, [-100, -100, -40])
correction['NoCorrection'] = fuzz.trimf(correction.universe, [-30, 0, 30])
correction['StrongRight'] = fuzz.trimf(correction.universe, [40, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(tilt['Left'] & velocity['Negative'], correction['StrongRight'])
rule2 = ctrl.Rule(tilt['Left'] & velocity['Zero'], correction['StrongRight'])
rule3 = ctrl.Rule(tilt['Left'] & velocity['Positive'], correction['StrongRight'])

rule4 = ctrl.Rule(tilt['Center'] & velocity['Zero'], correction['NoCorrection'])

rule5 = ctrl.Rule(tilt['Right'] & velocity['Negative'], correction['StrongLeft'])
rule6 = ctrl.Rule(tilt['Right'] & velocity['Zero'], correction['StrongLeft'])
rule7 = ctrl.Rule(tilt['Right'] & velocity['Positive'], correction['StrongLeft'])

# Create control system
system = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4,
    rule5, rule6, rule7
])

simulation = ctrl.ControlSystemSimulation(system)

# Take user input
t = float(input("Enter tilt angle: "))
v = float(input("Enter angular velocity: "))

# Pass inputs
simulation.input['tilt'] = t
simulation.input['velocity'] = v

# Calculate output
simulation.compute()

# Display result
print("Drone Correction:", simulation.output['correction'])