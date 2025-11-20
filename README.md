# ⚛️ QLCM-Qiskit – Experimental Implementation  
**Quantum Language & Consciousness Model**  
*Validated with real Swap-Test + FakeLima noise*

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17565578.svg)](https://doi.org/10.5281/zenodo.17565578)  
**Author:** Osmary Lisbeth Navarro Tovar  
**Lab:** Quantum Communication & Consciousness Laboratory  
**Paper:** https://ccuantica.com/qlcm/

---

## 🔑 Experimental Results (n = 84, p &lt; 0.001)

| Metric | QLCM | Control (Haar) | Gain |
|--------|------|----------------|------|
| **Semantic Coherence** *H&lt;sub&gt;s&lt;/sub&gt;* | 0.913 ± 0.047 | 0.412 ± 0.109 | +121 % |
| **Inter-Logon Overlap** *H&lt;sub&gt;IL&lt;/sub&gt;* | 0.87 ± 0.03 | 0.50 ± 0.08 | +74 % |
| **IQC** | 91.2 ± 2.1 | 47.3 ± 5.0 | +92 % |
| **Backend** | *FakeLima* (realistic noise) | *Haar* (random) | — |

---

## 🔬 What is QLCM-Qiskit?

**QLCM-Qiskit** is the **experimental implementation** of the *Quantum Language & Consciousness Model* (QLCM), a framework that treats **language as a quantum field of conscious information**.

### Core Features:
- **Logon Creation**: Quantum states representing semantic quanta (1D-2D-3D auto-dimensioned)
- **Real Swap-Test**: Measures **|⟨ψ|φ⟩|²** under **FakeLima noise**
- **Reproducible**: Fixed seed + pinned dependencies
- **Exportable**: CSV + PNG outputs ready for papers

---

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `logon.py` | **QLCM Core Class** – Creates Logons with real 3-qubit circuits |
| `coherencia.py` | **Inter-Logon Correlation** – **Swap-Test real** + FakeLima noise |
| `experimento.ipynb` | **PoC Notebook** – Run experiment, export CSV/PNG |
| `requirements.txt` | **Pinned versions** – Reproducible stack |

---

## 🚀 Quick Start (1 minute)

```bash
git clone https://github.com/ccuantica/QLCM-Qiskit.git
cd QLCM-Qiskit
python -m venv qlcm-env
source qlcm-env/bin/activate
pip install -r requirements.txt
jupyter lab experimento.ipynb


📊 What does the notebook do?
Creates 1D-2D-3D Logons with real quantum circuits
Measures H<sub> via Swap-Test under FakeLima noise
Exports qlcm_correlacion_auto_realtime.csv + qlcm_correlacion_vs_tiempo_auto.png
Reproducible: seed 42 + pinned dependencies
📝 Citation (BibTeX)
bibtex
Copy
@software{navarro_tovar_qlcm-qiskit_2025,
  author       = {Navarro Tovar, Osmary Lisbeth},
  title        = {{QLCM-Qiskit: Quantum Language \& Consciousness Model v1.0v}},
  month        = nov,
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.0v},
  doi          = {10.5281/zenodo.17565578},
  url          = {https://doi.org/10.5281/zenodo.17565578},
  note         = {MIT License}
}
🔓 License
MIT © 2025 Osmary Lisbeth Navarro Tovar
Use it, modify it, cite it.
