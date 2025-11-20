#!/usr/bin/env python3
"""
QLCM v1.0v – Logon Core (drop-in)
Same API, real-quantum interior.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeLimaV2
from qiskit.quantum_info import Statevector, state_fidelity, state_tomography
from datetime import datetime
import time
import random

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------- SEED & BACKEND ----------
np.random.seed(42)
random.seed(42)
BACKEND = FakeLimaV2()

# ---------- INTERNAL UTILS ----------
def _build_3q_circuit(θs: float, θa: float, θe: float) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.ry(θs, 0); qc.ry(θa, 1); qc.ry(θe, 2)
    qc.cx(0, 1); qc.cx(1, 2)
    return qc

def _noisy_rho(qc: QuantumCircuit):
    qc_t = transpile(qc, BACKEND)
    tomo = state_tomography.StateTomography(qc_t, [0, 1, 2])
    return tomo.fit()

def _eth_proj(rho):
    evals, evecs = np.linalg.eigh(rho)
    mask = evals < 0.2
    return evecs[:, mask] @ evecs[:, mask].T.conj()

def _coherence(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return 1 - -np.sum(evals * np.log2(evals))

def _resonance(rho):
    op = np.kron(np.eye(2), np.kron([[0, 0], [0, 1]], np.eye(2)))
    return float(np.real(np.trace(rho @ op)))

# ---------- PUBLIC CLASS ----------
class QuantumLogonAutoDimension:
    def __init__(self, label: str, semantic_vec: np.ndarray, affective_amp: float, intention_phase: float):
        self.label = label
        self.A_a = np.clip(affective_amp, 0.0, 1.0)
        self.φ_i = intention_phase

        # Auto-dimension: only 8 supported in v1.0v
        self.dimension_qubits = 3
        if semantic_vec.size != 8:
            raise ValueError("v1.0v only supports dim=8 (3 qubits).")

        # Map amplitudes → Ry angles
        θs = np.abs(semantic_vec[0]) * np.pi
        θa = self.A_a * np.pi
        θe = (self.φ_i / np.pi) % 1 * np.pi

        qc = _build_3q_circuit(θs, θa, θe)
        self._rho_ideal = Statevector(qc).density_matrix()
        self._rho_real    = _noisy_rho(qc)

        self.metrics = {}

    def calculate_full_metrics(self, target_state=None, ethical_projector=None):
        # Hs (Semantic Fidelity)
        hs = state_fidelity(self._rho_real, self._rho_ideal)
        # Ef (Ethical Fidelity)
        ef = float(np.real(np.trace(self._rho_real @ _eth_proj(self._rho_real))))
        # IQC (Quantum Consciousness Coefficient)
        iqc = (0.4 * hs + 0.3 * self.A_a + 0.3 * ef) * 100.0
        # Real coherence and resonance
        coh = _coherence(self._rho_real)
        res = _resonance(self._rho_real)

        self.metrics = {
            'Hs': hs, 'Ef': ef, 'IQC': iqc,
            'coherence': coh, 'resonance': res,
            'timestamp': datetime.now().isoformat()
        }
        logging.info(f"Logon {self.label} | IQC: {iqc:.2f} | Dim: {self.dimension_qubits} Qubits")
        return self.metrics

    def export_csv(self, filename="qlcm_logon_instance.csv"):
        df = pd.DataFrame([{
            "timestamp": self.metrics['timestamp'],
            "label": self.label,
            "dimension_qubits": self.dimension_qubits,
            "IQC": self.metrics['IQC'],
            "Hs": self.metrics['Hs'],
            "Ef": self.metrics['Ef'],
            "affective_amp": self.A_a,
            "intention_phase": self.φ_i,
            "coherence": self.metrics['coherence'],
            "resonance": self.metrics['resonance']
        }])
        df.to_csv(filename, index=False)
        logging.info(f"CSV saved: {filename}")

# ---------- PLOTTING ----------
def plot_iqc_vs_time(metrics_list):
    times = [m["timestamp"] for m in metrics_list]
    iqc_values = [m["IQC"] for m in metrics_list]
    coh_values = [m["coherence"] for m in metrics_list]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(times, iqc_values, marker='o', color='#ff00ff', linewidth=2)
    plt.axhline(85.0, color='orange', linestyle='--', label='IQC Optimal Threshold')
    plt.xlabel("Time"); plt.ylabel("IQC"); plt.title("QLCM: IQC vs. Time"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(times, coh_values, marker='o', color='#00ff00', linewidth=2)
    plt.axhline(0.9, color='cyan', linestyle='--', label='Coherence Optimal Threshold')
    plt.xlabel("Time"); plt.ylabel("Coherence"); plt.title("QLCM: Coherence vs. Time"); plt.legend()
    plt.tight_layout(); plt.savefig("qlcm_iqc_vs_time.png", dpi=300, bbox_inches="tight"); plt.show()

def export_csv_batch(metrics_list, filename="qlcm_logon_batch_realtime.csv"):
    pd.DataFrame(metrics_list).to_csv(filename, index=False)
    logging.info(f"Batch CSV saved: {filename}")

# ---------- DEMO ----------
if __name__ == "__main__":
    print("🔬 QLCM Logon Auto-Dimension – Demo v1.0v (quantum core)")
    metrics_list = []
    for i in range(10):
        dim = 8
        state = np.random.rand(dim) + 1j * np.random.rand(dim)
        state /= np.linalg.norm(state)

        logon = QuantumLogonAutoDimension(
            label=f"Logon_{i}",
            semantic_vec=state,
            affective_amp=np.random.beta(8, 2),
            intention_phase=np.random.beta(7, 3)
        )
        metrics = logon.calculate_full_metrics()
        metrics_list.append(metrics)
        logon.export_csv(f"qlcm_logon_instance_{i}.csv")
        time.sleep(0.5)

    plot_iqc_vs_time(metrics_list)
    export_csv_batch(metrics_list)
    print("✅ Demo v1.0v finished – files generated.")
