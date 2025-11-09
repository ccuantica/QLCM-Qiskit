import numpy as np
from qiskit.quantum_info import Statevector

class Logon:
    def __init__(self, ν_s, A_a, φ_i):
        self.ν_s = ν_s      # frecuencia semántica
        self.A_a = A_a      # amplitud afectiva
        self.φ_i = φ_i      # fase intencional
    
    def to_statevector(self):
        alpha = np.sqrt(self.A_a) * np.exp(1j * self.φ_i)
        beta = np.sqrt(1 - self.A_a)
        return Statevector([alpha, beta])
