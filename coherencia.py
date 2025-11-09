from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeLima
import numpy as np

def calcular_Hs(logon1, logon2, shots=8192):
    qc = QuantumCircuit(2, 2)
    qc.initialize(logon1.to_statevector().data, 0)
    qc.initialize(logon2.to_statevector().data, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0,1], [0,1])
    
    backend = AerSimulator.from_backend(FakeLima())  # ruido real ibmq_lima
    qc = transpile(qc, backend)
    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()
    
    correlaciones = counts.get('00', 0) + counts.get('11', 0)
    return correlaciones / shots
