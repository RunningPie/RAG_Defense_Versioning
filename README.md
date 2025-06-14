# Version-Differential Cross-Referencing for RAG Defense

This repository contains the source code and experimental setup for the research paper:

**"Version-Differential Cross-Referencing Knowledge Base for RAG-based LLM Systems to Minimize Neighbor-Borrowing Item Demotion Attacks."**

The project investigates and implements a novel defense mechanism to protect Retrieval-Augmented Generation (RAG) systems from semantic data poisoning.

---

## 🔍 Overview

The core idea is to detect malicious updates to a knowledge base by:

- Analyzing changes between document versions
- Cross-referencing newly added text against the existing corpus
- Identifying suspicious content borrowing that may signal an attack

---

## 📈 Features

- **Attack Simulation:** Script to simulate "neighbor borrowing" data poisoning (Nazary et al.)
- **Defense Mechanism:** Modular implementation of Version-Differential Cross-Referencing
- **RAG Pipeline:** Complete RAG implementation with dense retriever and generator LLM
- **Experiment Orchestration:** Main script to run baseline, attacked, and defended evaluations
- **Configurable Parameters:** All settings in a single `config.yaml`
- **Jupyter Notebooks:** For data exploration, prototyping, and result visualization

---

## 📁 Project Structure

```
.
├── config.yaml                 # Central configuration for all parameters
├── data/
│   ├── processed/              # Final clean and poisoned knowledge bases
│   └── raw/                    # Original MovieLens dataset
├── main.py                     # Main experiment script
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_prototype_defense.ipynb
│   └── 3_visualize_results.ipynb
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── src/
    ├── attack/
    │   └── neighbor_borrowing.py     # Attack simulation
    ├── defense/
    │   └── version_diff.py           # Defense logic
    ├── evaluation/
    │   └── metrics.py                # MRR, HR@k metrics
    ├── rag_components/
    │   ├── generator.py              # LLM generation
    │   └── retriever.py              # Dense retriever
    └── pipeline.py                   # RAG system orchestration
```

---

## 🚀 Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

- `pandas`
- `groq`
- `sentence-transformers`
- `diff-match-patch`

### 4. Set Up API Key

Set your Groq API key as an environment variable:

```bash
# Windows
$env:GROQ_API_KEY="your_api_key_here"

# macOS/Linux
export GROQ_API_KEY="your_api_key_here"
```

Alternatively, you can hardcode it in `config.yaml` (not recommended).

### 5. Download MovieLens Dataset

Download [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/) and extract contents (e.g., `movies.csv`, `ratings.csv`) into:

```
data/raw/
```

---

## 🌍 How to Run the Experiments

Run all commands from the root directory.

### Step 1: Build Clean Knowledge Base

```bash
python -m src.data_processing.build_kb
```

Generates:

```
data/processed/clean_knowledge_base.json
```

### Step 2: Simulate the Attack

```bash
python -m src.attack.neighbor_borrowing
```

Generates:

```
data/processed/poisoned_knowledge_base.json
```

### Step 3: Run the Full Evaluation

```bash
python main.py
```

This script will:

- Evaluate the **baseline**, **attacked**, and **defended** versions
- Compute metrics (MRR, HR\@k)
- Save results in `results/` with a timestamped JSON file

---

## 📍 Acknowledgements

This project was made possible by the contributions of several open-source tools and data providers:

- **Dataset:** MovieLens `ml-latest-small`, by [GroupLens](https://grouplens.org/).
  - F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM TiiS 5, 4.
- **LLM Inference:** Powered by [Groq](https://groq.com/) API
- **Embeddings:** [sentence-transformers](https://www.sbert.net/)
- **Text Diffing:** [diff-match-patch](https://github.com/google/diff-match-patch) by Google

---

