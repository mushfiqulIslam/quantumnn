import itertools

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

from pipeline.hooks import circuit_outputs


class Quantum4QBitCircuit:

    def __init__(self, backend, shots):
        self.n_qubits = 4
        self._circuit = QuantumCircuit(self.n_qubits)

        # Hadamard gate ref: https://www.quantum-inspire.com/kbase/hadamard/
        all_qubits = [i for i in range(self.n_qubits)]
        self._circuit.h(all_qubits)

        # preparing and adding 4 parameters for 4 circuits
        self._parameters = []
        for i in range(self.n_qubits-1, -1, -1):
            theta_x = Parameter('theta-x{}'.format(i))
            theta_y = Parameter('theta-y{}'.format(i))
            theta_z = Parameter('theta-z{}'.format(i))
            self._circuit.rx(theta_x, [i])
            self._circuit.ry(theta_y, [i])
            self._circuit.rz(theta_z, [i])
            self._parameters = self._parameters + [theta_x, theta_y, theta_z]

        self._circuit.measure_all()

        self._backend = backend
        self._shots = shots

    def draw(self):
        self._circuit.draw("mpl", filename='M:/quntumnn/figures/{}-qubit circuit ryN.jpg'.format(self.n_qubits))
        print('Printed on M:/quntumnn/figures/{}-qubit circuit ryN.jpg'.format(self.n_qubits))

    def calculate_expectation(self, counts):
        circuit_output = circuit_outputs(self.n_qubits)
        expectations = np.zeros(len(circuit_output))
        for i in range(len(circuit_output)):
            key = circuit_output[i]
            perc = counts.get(key, 0) / self._shots
            expectations[i] = perc
        return expectations

    def simulate(self, thetas):
        # added all inputs to circuit
        self._circuit.bind_parameters(
            {self._parameters[k]: thetas[k] for k in range(self.n_qubits*3)}
        )

        job = execute(
            self._circuit,
            self._backend,
            shots=self._shots,
        )

        result = job.result()
        counts = result.get_counts()
        return self.calculate_expectation(counts)


if __name__ == '__main__':
    simulator = Aer.get_backend('qasm_simulator')

    circuit = Quantum4QBitCircuit(simulator, 160)
    print('Expected value for rotation pi {}'.format(circuit.simulate(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])[0]))
    circuit.draw()
