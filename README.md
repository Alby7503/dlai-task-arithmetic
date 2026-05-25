# Spectrum-Guided Task Arithmetic: Layer-Wise Merging in RoBERTa

This repository contains the official implementation of **Spectrum-Guided Task Arithmetic**, a project for the **Deep Learning and Artificial Intelligence (DLAI) 2025/2026** course at Sapienza University of Rome.

## 📖 Overview
Model merging via **Task Arithmetic** offers a zero-shot, computationally efficient alternative to multi-task fine-tuning by linearly combining the task vectors of fine-tuned networks. However, merging models fine-tuned on orthogonal tasks (e.g., highly syntactic logic tasks vs. highly semantic emotion classification) often leads to **catastrophic weight interference**.

Drawing on the architectural feature hierarchy of Transformers—where lower layers process general syntax/linguistic structures and upper layers encode task-specific semantics—we propose a **Layer-Wise Merging** strategy:
* **Bottom Layers (L0–L5 & Embeddings):** Merged using Task Arithmetic ($\lambda_{Emo}=0.6, \lambda_{Logic}=0.4$) to build a robust shared multi-task trunk.
* **Top Layers (L6–L11):** Kept task-pure to act as specialized task "heads", preserving high-level reasoning and classification capabilities.

Our empirical results demonstrate that this bottom-up layer-wise architecture effectively mitigates interference, significantly outperforming global full-model merging.

---

## 📂 Project Structure
* **`evaluate_dataset.py`**: Baseline script implementing global full-model merging (merging L0–L11).
* **`evaluate_hybrid_final.py`**: Core script implementing our **proposed bottom-up merging** (L0–L5 & embeddings merged, L6–L11 pure).
* **`evaluate_hybrid_reverse.py`** & **`evaluate_reverse.py`**: Ablation studies implementing **reverse layer-wise merging** (L0–L5 pure, L6–L11 merged) to validate our structural hierarchy hypothesis.
* **`test_vram.py`**: Utility script to inspect GPU hardware and VRAM consumption before running pipelines.
* **`DLAI_report.pdf`**: The final, fully compiled 2-page project report containing theoretical formalizations and results.
* **`report/main.tex`**: The raw LaTeX source code for the project report.

---

## ⚙️ Environment Setup & Installation
This project leverages the modern and fast Python package manager [**`uv`**](https://github.com/astral-sh/uv).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Alby7503/dlai-task-arithmetic.git
   cd dlai-task-arithmetic
   ```
2. **Install dependencies and create virtual environment:**
   ```bash
   uv sync
   ```
   *This automatically generates the `.venv` and installs PyTorch (with CUDA 12.1 support), Transformers, Datasets, and Scikit-Learn.*

---

## 🚀 Execution & Usage

### ⚠️ Windows Console Portability Notice
Due to terminal emoji markers (e.g., `📊`, `🚀`) printed in execution reports, standard Windows shells (which use `cp1252` encoding by default) will throw a `UnicodeEncodeError`. To prevent this, always set the standard stream encoding to UTF-8:

* **On Windows PowerShell:**
  ```powershell
  $env:PYTHONIOENCODING="utf-8"; uv run <script_name>.py
  ```
* **On Linux / macOS:**
  ```bash
  uv run <script_name>.py
  ```

### 1. Run Hardware Diagnostics
```powershell
$env:PYTHONIOENCODING="utf-8"; uv run test_vram.py
```

### 2. Evaluate the Full Merge Baseline
```powershell
$env:PYTHONIOENCODING="utf-8"; uv run evaluate_dataset.py
```

### 3. Evaluate the Proposed Layer-Wise Model (L0–L5 Merged)
```powershell
$env:PYTHONIOENCODING="utf-8"; uv run evaluate_hybrid_final.py
```

### 4. Evaluate the Reverse Merging Ablation (L6–L11 Merged)
```powershell
$env:PYTHONIOENCODING="utf-8"; uv run evaluate_hybrid_reverse.py
```

---

## 📊 Experimental Results

Experiments were validated on an **NVIDIA GeForce GTX 1080 GPU (8.00 GB VRAM)** on the SNLI (NLI) and GoEmotions (Emotion classification) test datasets:

| Strategy | Layer Merging | NLI Acc. (SNLI) | Emotion Macro F1 (GoEmotions) |
| :--- | :---: | :---: | :---: |
| **Pure Task A (Emotions)** | *None* | - | **52.10** |
| **Pure Task B (NLI)** | *None* | **89.50%** | - |
| **Full Merge** (Baseline) | L0–L11 | 79.39% | 38.20 |
| **Bottom-Up (Proposed)** | **L0–L5 & Embeddings** | **88.35%** | **48.03** |
| **Reverse** (Ablation) | L6–L11 | 88.35% | 45.03 |

> [!NOTE]
> The ablation study shows a **3.0 point collapse** in Emotion classification F1-score when interference is introduced into the top semantic layers (48.03 $\rightarrow$ 45.03), while logical accuracy remains stable (88.35%). This empirically proves our hypothesis that upper-layer specialized parameters are highly fragile and must be preserved task-pure.