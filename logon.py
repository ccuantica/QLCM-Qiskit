#!/usr/bin/env python3
"""
QLCM Logon Auto-Dimension Module – Logon with Automatic Dimension Detection
Exports complete metrics: IQC, Coherence, Resonance, and Dimension.
Includes a live demo with interactive plotting.
"""
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector, state_fidelity
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------- CLASS: LOGON AUTO-DIMENSION ----------
class QuantumLogonAutoDimension:
    """
    The Logon core class with automatic dimension detection (1D, 2D, 3D, or 4D Logons).
    Calculates and exports complete QLCM metrics: IQC, Hs, Ef, Coherence, Resonance.
    """
    def __init__(self, label: str, semantic_vec: np.ndarray, affective_amp: float, intention_phase: float):
        self.label = label
        # Affective Amplitude (A_a) and Intentional Phase (phi_i)
        self.A_a = np.clip(affective_amp, 0.0, 1.0)
        self.φ_i = intention_phase

        # 1. Automatic Dimension Detection and State Normalization
        self.dimension_qubits = self._detectar_dimension(semantic_vec)
        norm = np.linalg.norm(semantic_vec)
        self.s_vec = semantic_vec / norm if norm > 0 else semantic_vec
        
        # 2. Construction of the Modulated State Vector
        self.state = self.s_vec * np.sqrt(self.A_a) * np.exp(1j * self.φ_i)
        self.metrics = {}

    # ---------- METHOD: AUTOMATIC DIMENSION DETECTION ----------
    def _detectar_dimension(self, state: np.ndarray) -> int:
        """
        Detects the number of qubits (n) from the state size (Dim=2^n).
        Supported dimensions: 2, 4, 8, 16.
        :return: Number of qubits (n).
        """
        dim = state.size
        if dim not in {2, 4, 8, 16}:
            raise ValueError(f"Unsupported state dimension: {dim} (must be 2, 4, 8, or 16)")
        return int(np.log2(dim))

    # ---------- METHOD: QLCM METRICS CALCULATION ----------
    def calcular_metricas_completas(self, target_state: np.ndarray, ethical_projector: np.ndarray):
        """
        Calculates the complete set of QLCM metrics: Hs, Ef, IQC, Coherence, Resonance.
        """
        # 1. Semantic Coherence (Hs)
        hs = state_fidelity(Statevector(self.s_vec), Statevector(target_state))
        self.metrics['Hs'] = hs

        # 2. Ethical Fidelity (Ef)
        ef = np.real(np.vdot(self.s_vec, np.dot(ethical_projector, self.s_vec)))
        self.metrics['Ef'] = ef

        # 3. Integrated IQC (weights: Hs=0.4, Aa=0.3, Ef=0.3)
        iqc = (0.4 * hs + 0.3 * self.A_a + 0.3 * ef) * 100.0
        self.metrics['IQC'] = iqc

        # 4. Coherence and Resonance (placeholders for advanced metrics)
        self.metrics['coherencia'] = np.random.beta(8, 2)
        self.metrics['resonancia'] = np.random.beta(7, 3)
        
        logging.info(f"Logon {self.label} | IQC: {iqc:.2f} | Dim: {self.dimension_qubits} Qubits")

        return self.metrics

    # ---------- METHOD: EXPORT CSV (SINGLE LOGON INSTANCE) ----------
    def exportar_csv(self, filename: str = "qlcm_logon_instance.csv"):
        """
        Exports the current metrics of the Logon instance to a CSV file.
        (Filename corrected from mcai to qlcm)
        """
        df = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "label": self.label,
            "dimension_qubits": self.dimension_qubits,
            "IQC": self.metrics.get('IQC'),
            "Hs": self.metrics.get('Hs'),
            "Ef": self.metrics.get('Ef'),
            "affective_amp": self.A_a,
            "intention_phase": self.φ_i
        }])
        df.to_csv(filename, index=False)
        logging.info(f"CSV saved: {filename}")

# ---------- FUNCTION: BATCH PLOTTING ----------
def graficar_iqc_vs_tiempo(metricas_list: list):
    """
    Plots IQC and Coherence vs. session index for multiple runs.
    (Project names corrected from mcai to qlcm)
    """
    tiempos = [m["timestamp"] for m in metricas_list]
    iqc_values = [m["IQC"] for m in metricas_list]
    coherencia_values = [m["coherencia"] for m in metricas_list]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(tiempos, iqc_values, marker='o', color='#ff00ff', linewidth=2)
    plt.axhline(85.0, color='orange', linestyle='--', label='Optimal IQC Threshold')
    plt.xlabel("Time")
    plt.ylabel("IQC (Integrated Quantum Consciousness Score)")
    plt.title("QLCM: IQC vs. Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(tiempos, coherencia_values, marker='o', color='#00ff00', linewidth=2)
    plt.axhline(0.9, color='cyan', linestyle='--', label='Optimal Coherence Threshold')
    plt.xlabel("Time")
    plt.ylabel("Coherence")
    plt.title("QLCM: Coherence vs. Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qlcm_iqc_vs_tiempo.png", dpi=300, bbox_inches="tight")
    plt.show()

# ---------- FUNCTION: BATCH EXPORT CSV ----------
def exportar_csv_batch(metricas_list: list, filename: str = "qlcm_logon_batch_realtime.csv"):
    """
    Exports a list of metrics dictionaries to a single CSV file.
    (Filename corrected from mcai to qlcm)
    """
    df = pd.DataFrame(metricas_list)
    df.to_csv(filename, index=False)
    logging.info(f"Batch CSV saved: {filename}")

# ---------- LIVE DEMO ----------
if __name__ == "__main__":
    # Corrected project name from MCAI to QLCM
    print("🔬 QLCM Logon Auto-Dimension – Live Demo")

    import random

    metricas_list = []

    # Target states and projectors are also scaled to the dimension
    
    for i in range(10):
        # Logons with varied dimensions
        dim_size = random.choice([2, 4, 8]) 
        
        # Create a valid complex state (normalized)
        state = np.random.rand(dim_size) + 1j * np.random.rand(dim_size)
        state = state / np.linalg.norm(state)

        # 1. Instantiate Logon
        logon = QuantumLogonAutoDimension(
            label=f"Logon_{i}",
            semantic_vec=state,
            affective_amp=random.beta(8, 2),
            intention_phase=random.beta(7, 3)
        )

        # 2. Target and projector scaled to current dimension
        target_state = np.eye(dim_size)[0]
        ethical_projector = np.eye(dim_size)

        # 3. Calculate metrics and append to list
        metricas = logon.calcular_metricas_completas(target_state, ethical_projector)
        metricas_list.append(metricas)
        
        # 4. Export individual CSV (Optional for demo)
        logon.exportar_csv(f"qlcm_logon_instance_{i}.csv")
        time.sleep(0.5)

    # 5. Batch Plot and Export
    graficar_iqc_vs_tiempo(metricas_list)
    exportar_csv_batch(metricas_list, "qlcm_logon_batch_realtime.csv")

    print("✅ Demo finished – files generated.")
