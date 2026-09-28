# Licensing considerations (not legal advice)

| Component | Licence (verify current terms) | Implication |
|---|---|---|
| Ultralytics YOLOv8 code and Ultralytics-provided pretrained weights | AGPL-3.0 (Ultralytics also sells an Enterprise licence) | Software that imports/derives from it and is distributed or offered over a network is generally expected to be AGPL-3.0 compatible, with source available. |
| **This repository (proposed)** | **AGPL-3.0** - `LICENSE` contains the official text | Chosen for compatibility with Ultralytics. **You are the copyright holder: confirm or replace it** (e.g. obtain an Ultralytics Enterprise licence if you need closed-source/commercial use). If you replace it, also update `CITATION.cff`. |
| Fine-tuned weights (`pcb_*.pt`) | Treat as derived from Ultralytics weights (AGPL-3.0) **and** subject to your training-data terms | Confirm before redistribution. |
| Training dataset | **Unknown - not in supplied materials** | Must be documented in `dataset/README.md`. Many public datasets restrict redistribution or commercial use. |
| Streamlit, PyTorch, OpenCV, Pillow, pandas, NumPy | Permissive licences (Apache-2.0 / BSD-style / HPND) - verify exact versions | Keep attribution/notices when redistributing. |
| Screenshots in `assets/` | Your own dashboard captures | The demo board image inside them comes from the dataset - confirm it may be shown publicly. |

Third-party images that were uploaded alongside the project (a design mock-up, a Chinese YOLOv8 GUI screenshot with a CSDN watermark, a Roboflow-workflow UI screenshot and two research-paper figures) are **not included** in this repository because they are not screenshots of this dashboard and are others' copyrighted material.
