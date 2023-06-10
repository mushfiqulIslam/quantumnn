import numpy as np
from torch import tensor, Tensor, cat
from torch.autograd import Function
from torch.nn import Module

from pipeline.quantum_circuit import Quantum4QBitCircuit


class HybridFunction(Function):

    @staticmethod
    def forward(ctx, input_values, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation = ctx.quantum_circuit.simulate(input_values[0].tolist())
        result = tensor([expectation])
        ctx.save_for_backward(input_values, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        shift = ctx.shift
        input_values, forward_expectation = ctx.saved_tensors

        input_values = input_values[0].tolist()
        gradients = Tensor()

        for i in range(len(input_values)):

            # shift i-th value to right and left
            shift_right = input_values.copy()
            shift_right[i] = shift_right[i] + shift
            shift_left = input_values.copy()
            shift_left[i] = shift_left[i] - shift

            # simulate this value on the circuit
            expectation_right = ctx.quantum_circuit.simulate(shift_right)
            expectation_left = ctx.quantum_circuit.simulate(shift_left)
            
            gradient = tensor([expectation_right]) - tensor([expectation_left])
            gradients = cat((gradients, gradient.float()))

        return (gradients.float() * grad_output.float()).T, None, None


class Hybrid(Module):

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self._quantum_circuit = Quantum4QBitCircuit(backend, shots)
        self._shift = shift

    def forward(self, input_values):
        return HybridFunction.apply(input_values, self._quantum_circuit, self._shift)