import itertools


def circuit_outputs(n_qubits):
    measurements = list(itertools.product([0, 1], repeat=n_qubits))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]