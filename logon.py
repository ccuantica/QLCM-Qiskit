#!/usr/bin/env python3
"""
QLCM v1.0v – Logon Core (drop-in)
Misma API, interior real-quantum.
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

# ---------- SEMILLA & BACKEND ----------
np.random.seed(42)
random.seed(42)
BACKEND = FakeLimaV2()

# ---------- UTILS INTERNOS (ocultos) ----------
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

# ---------- CLASE PÚBLICA (misma firma) ----------
class QuantumLogonAutoDimension:
    def __init__(self, label: str, semantic_vec: np.ndarray, affective_amp: float, intention_phase: float):
        self.label = label
        self.A_a = np.clip(affective_amp, 0.0, 1.0)
        self.φ_i = intention_phase

        # Auto-dimensión: solo 8 soportado en v1.0v
        self.dimension_qubits = 3
        if semantic_vec.size != 8:
            raise ValueError("v1.0v solo soporta dim=8 (3 qubits).")

        # Mapeamos amplitudes → ángulos Ry
        θs = np.abs(semantic_vec[0]) * np.pi
        θa = self.A_a * np.pi
        θe = (self.φ_i / np.pi) % 1 * np.pi

        qc = _build_3q_circuit(θs, θa, θe)
        self._rho_ideal = Statevector(qc).density_matrix()
        self._rho_real   = _noisy_rho(qc)

        self.metrics = {}

    def calcular_metricas_completas(self, target_state=None, ethical_projector=None):
        # Hs
        hs = state_fidelity(self._rho_real, self._rho_ideal)
        # Ef
        ef = float(np.real(np.trace(self._rho_real @ _eth_proj(self._rho_real))))
        # IQC
        iqc = (0.4 * hs + 0.3 * self.A_a + 0.3 * ef) * 100.0
        # Coherencia y resonancia reales
        coh = _coherence(self._rho_real)
        res = _resonance(self._rho_real)

        self.metrics = {
            'Hs': hs, 'Ef': ef, 'IQC': iqc,
            'coherencia': coh, 'resonancia': res,
            'timestamp': datetime.now().isoformat()
        }
        logging.info(f"Logon {self.label} | IQC: {iqc:.2f} | Dim: {self.dimension_qubits} Qubits")
        return self.metrics

    def exportar_csv(self, filename="qlcm_logon_instance.csv"):
        df = pd.DataFrame([{
            "timestamp": self.metrics['timestamp'],
            "label": self.label,
            "dimension_qubits": self.dimension_qubits,
            "IQC": self.metrics['IQC'],
            "Hs": self.metrics['Hs'],
            "Ef": self.metrics['Ef'],
            "affective_amp": self.A_a,
            "intention_phase": self.φ_i,
            "coherencia": self.metrics['coherencia'],
            "resonancia": self.metrics['resonancia']
        }])
        df.to_csv(filename, index=False)
        logging.info(f"CSV guardado: {filename}")

# ---------- PLOTTING (sin cambios) ----------
def graficar_iqc_vs_tiempo(metricas_list):
    tiempos = [m["timestamp"] for m in metricas_list]
    iqc_values = [m["IQC"] for m in metricas_list]
    coh_values = [m["coherencia"] for m in metricas_list]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tiempos, iqc_values, marker='o', color='#ff00ff', linewidth=2)
    plt.axhline(85.0, color='orange', linestyle='--', label='Umbral Óptimo IQC')
    plt.xlabel("Tiempo"); plt.ylabel("IQC"); plt.title("QLCM: IQC vs. Tiempo"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(tiempos, coh_values, marker='o', color='#00ff00', linewidth=2)
    plt.axhline(0.9, color='cyan', linestyle='--', label='Umbral Óptimo Coherencia')
    plt.xlabel("Tiempo"); plt.ylabel("Coherencia"); plt.title("QLCM: Coherencia vs. Tiempo"); plt.legend()
    plt.tight_layout(); plt.savefig("qlcm_iqc_vs_tiempo.png", dpi=300, bbox_inches="tight"); plt.show()

def exportar_csv_batch(metricas_list, filename="qlcm_logon_batch_realtime.csv"):
    pd.DataFrame(metricas_list).to_csv(filename, index=False)
    logging.info(f"Batch CSV guardado: {filename}")

# ---------- DEMO (misma entrada) ----------
if __name__ == "__main__":
    print("🔬 QLCM Logon Auto-Dimension – Demo v1.0v (núcleo cuántico)")
    metricas_list = []
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
        metricas = logon.calcular_metricas_completas()
        metricas_list.append(metricas)
        logon.exportar_csv(f"qlcm_logon_instance_{i}.csv")
        time.sleep(0.5)

    graficar_iqc_vs_tiempo(metricas_list)
    exportar_csv_batch(metricas_list)
    print("✅ Demo v1.0v finalizada – archivos generados.")
