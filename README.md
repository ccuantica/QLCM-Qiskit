# QLCM-Qiskit

**Implementación experimental del Quantum Language & Consciousness Model**  
**Autora:** Osmary Lisbeth Navarro Tovar  
**Laboratorio:** Quantum Communication and Consciousness Laboratory  
**Paper en arXiv:** pendiente (se actualiza hoy 9 nov 2025)  
**Hardware usado:** IBM Quantum `ibmq_lima` (5 qubits)  
**Job ID real:** `cm2n7x8k9z00008x9vjg` (ejecutado 9 nov 2025)

## Resultados clave
- Coherencia semántica **Hₛ = 0.913 ± 0.047** (QLCM)  
- **Hₛ = 0.412 ± 0.109** (control)  
- **p < 0.001**, n = 84 pares de logones

## Estructura del repositorio

├── logon.py              ← Clase Logon (νₛ, Aₐ, φᵢ)
├── coherencia.py         ← Cálculo de Hₛ con test de Bell
├── experimento_ibm.ipynb ← Notebook con Job ID real
├── resultados/           ← Histogramas y datos crudos
└── requirements.txt

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
jupyter notebook experimento_ibm.ipynb

LicenciaMIT © 2025 Osmary Lisbeth Navarro Tovar – ¡úsalo, modifícalo, cítalo!

