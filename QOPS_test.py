import random
import os
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler


def get_random_circuit(qubits:int):
    circuit = random_circuit(qubits,4,2,seed=42)
    return circuit

def get_compact_program_specification_Z(circuit:QuantumCircuit,shots=4000, simulator_type='statevector'):
    aer_sim = AerSimulator(method=simulator_type, device="GPU", blocking_enable=True, batched_shots_gpu=True)
    pass_manager = generate_preset_pass_manager(backend=aer_sim, optimization_level=3)
    circuits = []
    pauli_string_list = []
    pauli = "Z"*(circuit.num_qubits)
    qc = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
    qc.compose(circuit,inplace=True)
    qc.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    circuits.append(qc)
    pauli_string_list.append(pauli)

    isa_qc = pass_manager.run(circuits)
    with Session(backend=aer_sim) as session:
        sampler = Sampler(mode=session)
        results = sampler.run(isa_qc, shots=shots).result()
    counts = {}

    for paulistring, result in zip(pauli_string_list, results):
        counts[paulistring] = result.data.c.get_counts()
    return counts

def load_program(name,path):
    qc = QuantumCircuit.from_qasm_file("{}/{}".format(path,name))
    qc.remove_final_measurements()
    if len(qc.clbits) > 0:
        for i in range(len(qc.clbits)):
            qc.measure(i, i)
    else:
        qc.measure_all()
    qc.remove_final_measurements()
    return qc.copy()


def get_mutants(n,circuit,seed=1997):
    mutants = mutation(circuit)
    return mutants


def mutation(circuit):
    m1,m2,m3 = circuit.copy(),circuit.copy(),circuit.copy()
    m1.ry(0.1,m1.num_qubits - 1)
    m2.rz(0.1,m1.num_qubits - 1)
    m3.rx(0.1,m1.num_qubits - 1)
    return [m1,m2,m3]




def conditional_operation_on_binary(binary_string: str) -> QuantumCircuit:
    """
    Creates a quantum circuit that applies a quantum operation only when the qubits
    are in the specific state defined by the binary_string.

    Parameters:
        binary_string (str): The binary string to condition on (e.g., '101').
        operation (str): The single-qubit gate to apply ('x', 'z', 'h', etc.)

    Returns:
        QuantumCircuit: A quantum circuit with the conditional operation.
    """
    n = len(binary_string)
    qc = QuantumCircuit(n)

    # Flip qubits that should be |0⟩ so condition becomes all-1 for MCX
    for i, bit in enumerate(binary_string):
        if bit == '0':
            qc.x(i)

    # Choose the target qubit for the operation (use last qubit if not specified)
    target = n - 1
    controls = list(range(n))
    controls.remove(target)

    # Apply MCX to target a flag qubit if condition matches
    mcx = MCXGate(len(controls))
    qc.append(mcx, controls + [target])

    # Apply the desired operation on target *if* controls match
    qc.p(0.01,target)

    # Uncompute the X gates
    for i, bit in enumerate(binary_string):
        if bit == '0':
            qc.x(i)

    return qc

def extend_qubits(qubits,circuit):
    qc = QuantumCircuit(qubits)
    qc = qc.compose(circuit)
    return qc

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_Z_family(qubits):
    N = 2**qubits
    f = []
    for i in range(0,N):
        b = bin(i)[2:]
        b = b.zfill(qubits)
        b = b.replace("1","Z").replace("0","I")
        f.append(b)
    return f

def get_random_Z_family(qubits):
    N = 2**qubits
    total = range(1, random.randint(2, 32))
    values = [random.randint(a=1, b=N-1) for _ in total]
    f = []
    for i in values:
        b = bin(i)[2:]
        b = b.zfill(qubits)
        b = b.replace("1","Z").replace("0","I")
        f.append(b)
    return f

def get_Z_family_values(qubits,values):
    f = []
    for i in values:
        b = bin(i)[2:]
        b = b.zfill(qubits)
        b = b.replace("1","Z").replace("0","I")
        f.append(b)
    return f



if __name__ == '__main__':
    pass