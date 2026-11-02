import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input variables
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')

# Output variable
fanSpeed = ctrl.Consequent(np.arange(0, 1601, 1), 'fanSpeed')

# Temperature membership functions
temperature['TooCold'] = fuzz.trimf(temperature.universe, [0, 0, 10])
temperature['Cold'] = fuzz.trimf(temperature.universe, [5, 12, 20])
temperature['Warm'] = fuzz.trimf(temperature.universe, [15, 23, 30])
temperature['Hot'] = fuzz.trimf(temperature.universe, [25, 33, 40])
temperature['TooHot'] = fuzz.trimf(temperature.universe, [35, 50, 50])

# Humidity membership functions
humidity['Low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['Medium'] = fuzz.trimf(humidity.universe, [25, 50, 75])
humidity['High'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# Fan speed membership functions
fanSpeed['Slow'] = fuzz.trimf(fanSpeed.universe, [0, 0, 600])
fanSpeed['Medium'] = fuzz.trimf(fanSpeed.universe, [400, 800, 1200])
fanSpeed['Fast'] = fuzz.trimf(fanSpeed.universe, [1000, 1600, 1600])

# Rules
rule1 = ctrl.Rule(temperature['TooCold'], fanSpeed['Slow'])
rule2 = ctrl.Rule(temperature['Cold'] & humidity['Low'], fanSpeed['Slow'])
rule3 = ctrl.Rule(temperature['Warm'] & humidity['Medium'], fanSpeed['Medium'])
rule4 = ctrl.Rule(temperature['Hot'] & humidity['High'], fanSpeed['Fast'])
rule5 = ctrl.Rule(temperature['TooHot'], fanSpeed['Fast'])

# Controller
system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
simulation = ctrl.ControlSystemSimulation(system)

# User input
t = float(input("Enter temperature: "))
h = float(input("Enter humidity: "))

simulation.input['temperature'] = t
simulation.input['humidity'] = h

simulation.compute()

print("Fan Speed:", simulation.output['fanSpeed'], "RPM")