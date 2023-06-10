import numpy as np
from torch import tensor
from torch.autograd import Function
from torch.nn import Module

from pipeline.back_quantum_circuit import QuantumCircuit


class HybridFunction(Function):

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        #         print("expectation_z: ", expectation_z)
        result = tensor([expectation_z])
        #         print("result", result)
        ctx.save_for_backward(input, result)
        #         print("FORWARD END")
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input_value, expectation_z = ctx.saved_tensors

        input_list = np.array(input_value.tolist())
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = expectation_right - expectation_left
            gradients.append(gradient)

        gradients = np.array([gradients]).T

        return tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, q_bit, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(q_bit, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)