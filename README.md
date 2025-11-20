# ⚛️ QLCM-Qiskit: Quantum Language & Consciousness Model – Experimental Implementation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17565578.svg)](https://doi.org/10.5281/zenodo.17565578)
**Author:** Osmary Lisbeth Navarro Tovar
**Laboratory:** Quantum Communication and Consciousness Laboratory
**Paper/Full Theory:** https://ccuantica.com/qlcm/

---

## 🔑 Key Experimental Results

The framework successfully validated the difference between coherent Logon states and control states under realistic quantum noise:

* **Semantic Coherence (QLCM):** $H_s = 0.913 \pm 0.047$ 
* **Control Group:** $H_s = 0.412 \pm 0.109$ 
* **Significance:** $p < 0.001$, $n = 84$ pairs 
* **Validation Backend:** Simulated noise identical to **`ibmq_lima`** using Qiskit's `FakeLima()`

---

## 🔬 Description

**QLCM-Qiskit** is the experimental implementation of the **Quantum Language & Consciousness Model (QLCM)**, a theoretical framework that re-conceptualizes language as a quantum field of conscious information.

### Core Functionalities:
* **Logon Creation:** Create **Logons**—semantic quanta defined by a semantic vector ($\nu_s$), Affective Amplitude ($A_a$), and Intentional Phase ($\varphi_i$).
* **Auto-Dimensioning:** The code automatically detects the Logon dimension (1D, 2D, 3D Logons) and scales the quantum circuit accordingly.
* **Semantic Coherence Measurement ($H_s$):** Measure $H_s$ using quantum circuits based on Bell tests, validated under **realistic noise**.
* **Auditability:** Generate reproducible experiments using Qiskit simulators or real IBM Quantum hardware.

This project establishes a transdisciplinary bridge between science, art, and spirituality, positioning language as a fundamental conscious technology for the informational and perceptive co-creation of shared realities.

---

## 📂 Repository Structure

| File | Description | Function |
| :--- | :--- | :--- |
| `logon_auto_dimension.py` | **QLCM Core Class** | Defines the `QuantumLogonAutoDimension` class ($v_s, A_a, \varphi_i$). |
| `coherencia.py` | **Validation Module** | Calculates **Inter-Logon Correlation ($H_{IL}$)** and Semantic Coherence ($H_s$) using auto-dimensioned Bell tests on the `FakeLima` noise model. |
| `experimento.ipynb` | **Proof-of-Concept (PoC)** | Jupyter Notebook demonstrating Logon creation, $H_{IL}$ calculation under noise, and visualization. **This notebook is fully reproducible.** |
| `requirements.txt` | Dependency List | Lists required Python packages. |

---

## 🚀 How to Run the Experiment

### ⚡ Quick Installation
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ccuantica/QLCM-Qiskit.git](https://github.com/ccuantica/QLCM-Qiskit.git)
    cd QLCM-Qiskit
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Experiment:** Open the notebook in your environment.
    ```bash
    jupyter notebook experimento.ipynb
    ```

---

## 📝 Citation

If you use this work in your research, please cite the following Zenodo DOI:

```bibtex
@software{navarro_tovar_2025,
  author         = {Navarro Tovar, Osmary Lisbeth},
  title          = {ccuantica/QLCM-Qiskit: QLCM v1.0},
  month          = nov,
  year           = 2025,
  publisher      = {Zenodo},
  version        = {v1.0},

License: MIT © 2025 Osmary Lisbeth Navarro Tovar – Use it, modify it, cite it!
  doi            = {10.5281/zenodo.17565578},
  url            = {[https://doi.org/10.5281/zenodo.17565578](https://doi.org/10.5281/zenodo.17565578)}
}
