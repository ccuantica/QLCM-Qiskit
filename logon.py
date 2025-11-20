# logon.py
# Reference Implementation of the Logon (QLCM v1.0)
# DOI: 10.5281/zenodo.17565578

import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity

class QuantumLogon:
    """
    Fundamental unit of conscious information within the QLCM framework.
    Models the Logon as a state integrating semantics, affect, and ethics.
    """
    def __init__(self, label, semantic_vec, affective_amp, intention_phase):
        """
        Initializes the Logon state.
        :param semantic_vec: Base semantic concept vector (State |s>).
        :param affective_amp: Affective Amplitude (A_a) [0, 1].
        :param intention_phase: Intentional Phase (phi_i) [0, 2pi].
        """
        self.label = label
        
        # 1. Normalization of the base semantic vector
        norm = np.linalg.norm(semantic_vec)
        self.s_vec = semantic_vec / norm if norm > 0 else semantic_vec
        
        self.A_a = np.clip(affective_amp, 0.0, 1.0)
        self.φ_i = intention_phase
        self.metrics = {}

        # 2. Construction of the Vibrational State: 
        complex_coeff = np.sqrt(self.A_a) * np.exp(1j * self.φ_i)
        self.state = self.s_vec * complex_coeff

    # ----------------------------------------------------
    # OPERATIONAL METRICS CALCULATION METHODS
    # ----------------------------------------------------

    def compute_hs(self, target_state):
        """
        Calculates Semantic Coherence (H_s) vs. an Ideal Concept (|s_ideal>).
        """
        fid = state_fidelity(Statevector(self.s_vec), Statevector(target_state))
        self.metrics['Hs'] = fid
        return fid

    def compute_ef(self, ethical_projector):
        """
        Calculates Ethical Fidelity (E_f). Projection onto the ethical subspace.
        """
        expectation = np.vdot(self.s_vec, np.dot(ethical_projector, self.s_vec))
        self.metrics['Ef'] = np.real(expectation)
        return self.metrics['Ef']

    def calculate_iqc(self, w_s=0.4, w_a=0.3, w_e=0.3):
        """
        Calculates the Integrated Quantum Consciousness Score (IQC).
        """
        hs = self.metrics.get('Hs', 0.0)
        ef = self.metrics.get('Ef', 0.0)
        aa = self.A_a
        
        if not np.isclose(w_s + w_a + w_e, 1.0):
             print("Warning: IQC weights do not sum to 1.0")

        iqc = (w_s * hs + w_a * aa + w_e * ef) * 100.0
        self.metrics['IQC'] = iqc
        return iqc

# --- USAGE EXAMPLE (PoC Validation) ---
if __name__ == "__main__":
    print(f"--- QLCM PROTOCOL: Logonic Validation Test ---")
    
    state_ideal_unity = np.array([1, 0]) 
    op_ethical = np.array([[1, 0], [0, 0]]) 

    logon_integration = QuantumLogon(
        label="Integración", 
        semantic_vec=np.array([0.9, 0.1]),
        affective_amp=0.95, 
        intention_phase=0.0
    )
    
    hs = logon_integration.compute_hs(state_ideal_unity)
    ef = logon_integration.compute_ef(op_ethical)
    iqc = logon_integration.calculate_iqc()
    
    print(f"\nLogon: [{logon_integration.label}]")
    print(f" > IQC FINAL:      {iqc:.1f}/100")
    
    if iqc >= 85.0:
        print("\n[VALIDATED] Logon exceeds PoC Threshold (85.0).")
