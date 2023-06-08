import numpy as np
import qiskit


def exctract_single_qubit_measurment(dict_of_counts, qubit_range):
    num_qubits = len(list(dict_of_counts.keys())[0])
    result = np.zeros(len(qubit_range))
    result = np.zeros(num_qubits)
#     print(result)
    for el in dict_of_counts:
        for i in range(num_qubits):
            if i in qubit_range and el[i] == '1':
                result[i] += dict_of_counts[el]
                
    return result[qubit_range]


class QuantumCircuit:

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        # Hadamard gate ref: https://www.quantum-inspire.com/kbase/hadamard/
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = qiskit.execute(
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


if __name__ == '__main__':
    simulator = qiskit.Aer.get_backend('qasm_simulator')

    circuit = QuantumCircuit(1, simulator, 100)
    print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
    print(circuit._circuit.draw())