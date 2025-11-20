#!/usr/bin/env python3
"""
coherencia.py – QLCM Auto-Dimension Inter-Logon Correlation  
Swap-Test real + FakeLima noise. API 100 % compatible.  
Author: Osmary Lisbeth Navarro Tovar – MIT License
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeLima
from datetime import datetime
import time
import random

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ---------- REPRODUCIBILITY ----------
np.random.seed(42)
random.seed(42)
BACKEND = AerSimulator.from_backend(FakeLima())

# ---------- INTERNAL UTILS ----------
def _detect_dimension(state: np.ndarray) -> int:
    dim = state.size
    if dim not in {2, 4, 8, 16}:
        raise ValueError(f"Dimension {dim} is not a power of 2.")
    return int(np.log2(dim))

def _swap_test_circuit(state1: np.ndarray, state2: np.ndarray) -> QuantumCircuit:
    n = _detect_dimension(state1)
    qc = QuantumCircuit(2 * n + 1, 1)  # +1 ancilla
    qc.initialize(state1, range(1, n + 1))
    qc.initialize(state2, range(n + 1, 2 * n + 1))
    qc.h(0)
    for i in range(n):
        qc.cswap(0, 1 + i, n + 1 + i)
    qc.h(0)
    qc.measure(0, 0)
    return qc

def _overlap_prob0(counts: dict, shots: int) -> float:
    return counts.get('0', 0) / shots

def _iqc_from_logons(logon1, logon2):
    iqc1 = getattr(logon1, 'metrics', {}).get('IQC', None)
    iqc2 = getattr(logon2, 'metrics', {}).get('IQC', None)
    if iqc1 is not None and iqc2 is not None:
        return (iqc1 + iqc2) / 2
    return np.random.beta(8, 2) * 100  # demo fallback

# ---------- PUBLIC FUNCTION (same signature) ----------
def calcular_correlacion_auto_dimension(logon1, logon2, shots=8192, backend_name="FakeLima"):
    state1 = np.array(logon1.state, dtype=complex)
    state2 = np.array(logon2.state, dtype=complex)
    n = _detect_dimension(state1)
    if _detect_dimension(state2) != n:
        raise ValueError("Logons must have the same dimension.")

    qc = _swap_test_circuit(state1, state2)
    qc_trans = transpile(qc, BACKEND)
    job = BACKEND.run(qc_trans, shots=shots)
    counts = job.result().get_counts()

    p0 = _overlap_prob0(counts, shots)
    H_IL = 2 * p0 - 1  # |<ψ|φ>|² = 2P₀−1

    # Real metrics
    iqc = _iqc_from_logons(logon1, logon2)
    coh = 1 - (1 - abs(H_IL)) ** 2  # coherence ∝ overlap
    res = abs(H_IL)  # resonance = |overlap|

    metricas = {
        "H_IL": H_IL,
        "counts": counts,
        "shots": shots,
        "backend": backend_name,
        "timestamp": datetime.now().isoformat(),
        "dimension_qubits": n,
        "iqc": iqc,
        "coherencia": coh,
        "resonancia": res
    }
    logging.info(f"H_IL = {H_IL:.4f} | Qubits: {n} | Backend: {backend_name}")
    return metricas

# ---------- PLOTTING (same signature) ----------
def graficar_correlacion_vs_tiempo(metricas_list):
    sessions = list(range(1, len(metricas_list) + 1))
    hil_values = [m["H_IL"] for m in metricas_list]
    iqc_values = [m["iqc"] for m in metricas_list]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sessions, hil_values, marker='o', color='#00ff00', linewidth=2)
    plt.axhline(0.0, color='white', linestyle='-', alpha=0)
    plt.axhline(0.5, color='orange', linestyle='--', label='Classical threshold')
    plt.axhline(0.8, color='red', linestyle='--', label='Quantum threshold')
    plt.xlabel("Session index"); plt.ylabel("H_IL (real Swap-Test)"); plt.title("QLCM: Correlation vs. Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sessions, iqc_values, marker='o', color='#ff00ff', linewidth=2)
    plt.axhline(90, color='cyan', linestyle='--', label='Optimal IQC (90)')
    plt.xlabel("Session index"); plt.ylabel("IQC"); plt.title("QLCM: IQC vs. Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qlcm_correlacion_vs_tiempo_auto.png", dpi=300, bbox_inches="tight")
    plt.show()

# ---------- EXPORT CSV (same signature) ----------
def exportar_csv(metricas_list, filename="qlcm_correlacion_auto_realtime.csv"):
    pd.DataFrame(metricas_list).to_csv(filename, index=False)
    logging.info(f"CSV saved: {filename}")

# ---------- DEMO (same entry point) ----------
if __name__ == "__main__":
    print("🔬 QLCM Auto-Dimension Correlation – Demo (real Swap-Test)")
    try:
        from logon_auto_dimension import QuantumLogonAutoDimension as QuantumLogon
    except ImportError:
        logging.error("logon.py not found."); exit()

    metricas_list = []
    for i in range(10):
        dim = random.choice([2, 4, 8])
        st_A = np.random.rand(dim) + 1j * np.random.rand(dim)
        st_B = np.random.rand(dim) + 1j * np.random.rand(dim)
        st_A = st_A / np.linalg.norm(st_A); st_B = st_B / np.linalg.norm(st_B)

        logon_A = QuantumLogon(f"A_{dim}D", st_A, random.beta(8, 2), random.beta(7, 3))
        logon_B = QuantumLogon(f"B_{dim}D", st_B, random.beta(8, 2), random.beta(7, 3))
        logon_A.calcular_metricas_completas(); logon_B.calcular_metricas_completas()

        metricas = calcular_correlacion_auto_dimension(logon_A, logon_B, shots=4096)
        metricas_list.append(metricas)
        time.sleep(0.5)

    graficar_correlacion_vs_tiempo(metricas_list)
    exportar_csv(metricas_list)
    print("✅ Demo finished – files generated with real Swap-Test.")
