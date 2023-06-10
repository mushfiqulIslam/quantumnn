import numpy as np
from qiskit import QuantumCircuit as BaseQuantumCircuit
from qiskit import execute, Aer
from qiskit.circuit import Parameter


def exctract_single_qubit_measurment(dict_of_counts, qubit_range):
    print(dict_of_counts, qubit_range)
    num_qubits = len(list(dict_of_counts.keys())[0])
    print(num_qubits)
    result = np.zeros(num_qubits)

    for el in dict_of_counts:
        for i in range(num_qubits):
            if i in qubit_range and el[i] == '1':
                print(el, el[i], dict_of_counts[el])
                result[i] += dict_of_counts[el]

    return result[qubit_range]


class QuantumCircuit:

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = BaseQuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        all_qubits = [i for i in range(n_qubits)]
        self.theta = Parameter('theta')

        # Hadamard gate ref: https://www.quantum-inspire.com/kbase/hadamard/
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = execute(
            self._circuit.bind_parameters({self.theta: thetas[0]}),
            self.backend,
            shots=self.shots,

        )
        result = job.result().get_counts()
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        new_probabilities = exctract_single_qubit_measurment(result, list(range(self.n_qubits))) / self.shots
        return new_probabilities

    def draw(self):
        print(self._circuit.draw("mpl", filename='M:/quntumnn/figures/{}-qubit circuit ryN.jpg'.format(self.n_qubits)))


if __name__ == '__main__':
    simulator = Aer.get_backend('qasm_simulator')

    circuit = QuantumCircuit(10, simulator, 100)
    print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
    circuit.draw()