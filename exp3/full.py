# Experiment 3B: Full Adder using MCP Neurons

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


# OR Gate using MCP
def or_gate(a, b):
    return mcp_neuron([a, b], [1, 1], 1)


# XOR Gate
def xor_gate(a, b):
    return (a and not b) or (not a and b)


# Full Adder
def full_adder(a, b, cin):

    # Calculate Sum
    sum_bit = int(xor_gate(xor_gate(a, b), cin))

    # Calculate Carry
    carry1 = and_gate(a, b)
    carry2 = and_gate(a, cin)
    carry3 = and_gate(b, cin)

    carry = or_gate(carry1, or_gate(carry2, carry3))

    return sum_bit, carry


# Main Program
print("FULL ADDER")
print("A  B  Cin  SUM  CARRY")

for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:

            sum_bit, carry = full_adder(a, b, cin)

            print(a, b, cin, " ", sum_bit, "   ", carry)