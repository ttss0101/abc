# Experiment 3A: Half Adder using MCP Neuron

# MCP Neuron
def mcp_neuron(inputs, weights, threshold):

    net = 0

    for i in range(len(inputs)):
        net += inputs[i] * weights[i]

    if net >= threshold:
        return 1
    else:
        return 0


# AND Gate using MCP
def and_gate(a, b):
    return mcp_neuron([a, b], [1, 1], 2)


# XOR Gate
def xor_gate(a, b):
    return (a and not b) or (not a and b)


# Half Adder
def half_adder(a, b):

    sum_bit = int(xor_gate(a, b))

    carry = and_gate(a, b)

    return sum_bit, carry


# Main Program
print("HALF ADDER")
print("A  B  SUM  CARRY")

for a in [0, 1]:
    for b in [0, 1]:

        sum_bit, carry = half_adder(a, b)

        print(a, b, " ", sum_bit, "   ", carry)