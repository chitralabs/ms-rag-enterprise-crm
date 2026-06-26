# Multi-Source RAG Framework for Intelligent Agent Orchestration in Enterprise CRM Systems

**Paper:** "A Multi-Source Retrieval-Augmented Generation Framework for Intelligent Agent Orchestration in Enterprise CRM Systems"  
**Venue:** IEEE Access, 2026  
**Author:** Chitrapradha Ganesan (Senior Member, IEEE) — The University of Texas at Austin  
**Contact:** chitracrmexpert@gmail.com | ORCID: 0009-0009-1305-1724

---

> **Disclaimer:** This repository contains reproducibility materials for a public-dataset-based academic study. It does not contain proprietary data, employer data, customer records, confidential CRM configurations, internal system information, or commercial deployment metrics. All experiments are designed around publicly available datasets.

---

## Overview

Enterprise CRM environments require query resolution across heterogeneous knowledge sources—structured knowledge articles, CMS repositories, unstructured PDFs, and cloud-hosted files. Standard single-source RAG systems are insufficient for this setting.

**MS-RAG** is a five-layer multi-source Retrieval-Augmented Generation framework that addresses:
- Heterogeneous multi-source ingestion with schema normalization
- Hybrid sparse+dense retrieval with RRF rank fusion
- Deduplication and top-k passage selection before generation
- Access-controlled dual-agent orchestration (authenticated vs. public portal users)
- Confidence-based escalation to human support queues
- Event-driven (Kafka-style) synchronization to keep indexes current

---

## Architecture Summary

```
┌────────────────────────────────────────────────────────────┐
│              Multi-Source Ingestion Layer                  │
│  Knowledge Articles | CMS | PDFs | Cloud Files            │
│  Schema normalization · Chunking · Metadata tagging        │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│              Hybrid Retrieval Layer                        │
│  BM25 (sparse) + Bi-encoder dense (FAISS)                 │
│  → Reciprocal Rank Fusion (RRF)                           │
│  → SHA-256 deduplication → top-m passage selection        │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│              LLM Grounding & Generation Layer              │
│  Structured prompt templates with retrieval constraints    │
│  Hallucination mitigation via source attribution           │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│              Dual-Agent Orchestration Layer                │
│  Access Router: authenticated vs. public CRM users        │
│  Confidence-based escalation → human support queue        │
└───────────────────────────┬────────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────────┐
│              Kafka-Style Synchronization Layer             │
│  Event-driven index updates from connected data sources   │
│  Source-specific sync modes (push / poll / webhook)       │
└────────────────────────────────────────────────────────────┘
```

---

## Public Datasets Used

| Dataset | Description | Official Link |
|---------|-------------|---------------|
| **MS MARCO** Passage Ranking | Large-scale passage retrieval benchmark (web queries) | https://microsoft.github.io/msmarco/ |
| **MSDialog** | Conversational QA from Microsoft community forums | https://ciir.cs.umass.edu/downloads/msdialog/ |
| **WixQA** | Enterprise customer-support KB with article-level relevance labels | https://github.com/wix-incubator/WixQA |

See [`data/README.md`](data/README.md) for download instructions.

---

## Key Results (Public-Dataset Ablation)

### MS MARCO (hybrid vs. baselines)
| Configuration | ROUGE-L | MRR | P@5 |
|--------------|---------|-----|-----|
| BM25-only | 0.371 | 0.598 | 0.541 |
| Dense-only | 0.408 | 0.641 | 0.597 |
| **MS-RAG Hybrid** | **0.429** | **0.672** | **0.630** |

All differences p < 0.001, paired Wilcoxon test. Retrieval pipeline latency: 274 ms.

### WixQA (enterprise KB)
| Configuration | MRR |
|--------------|-----|
| BM25-only | 0.340 |
| Dense-only | 0.499 |
| **MS-RAG Hybrid** | **0.501** |

### Hallucination Rate (500 CRM-style queries, human-annotated)
| Configuration | Hallucination Rate |
|--------------|-------------------|
| LLM-only (no retrieval) | 34.2% |
| BM25-grounded | 14.1% |
| Dense-grounded | 9.3% |
| **MS-RAG Hybrid** | **7.8%** |

Inter-annotator agreement: κ = 0.81. Relative reduction vs. LLM-only: **77.2%**.

> **Note:** LLM generation results may vary unless the exact model snapshot, temperature, and decoding settings are matched. See [`configs/prompt_config.yaml`](configs/prompt_config.yaml) for the settings used in the paper.

---

## Repository Structure

```
ms-rag-enterprise-crm/
├── src/
│   ├── retrieval/          # BM25, dense retrieval, RRF fusion, deduplication
│   ├── orchestration/      # Access router, confidence escalation
│   ├── generation/         # Prompt templates
│   └── evaluation/         # ROUGE-L, MRR, P@k, hallucination metrics
├── scripts/                # Ablation and evaluation runner scripts
├── configs/                # YAML configuration files
├── data/                   # README with dataset download instructions only
├── results/                # Summary CSV results (public data only)
├── annotations/            # Hallucination annotation guidelines + anonymized sample
└── docs/                   # Architecture, reproducibility checklist, limitations
```

---

## Setup

### Requirements
- Python 3.9+
- See [`requirements.txt`](requirements.txt) and [`environment.yml`](environment.yml)

### Install

```bash
# Clone the repository
git clone https://github.com/chitralabs/ms-rag-enterprise-crm.git
cd ms-rag-enterprise-crm

# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate ms-rag
```

### Download Datasets

Follow instructions in [`data/README.md`](data/README.md).

### Configure

Copy and edit the configs:
```bash
cp configs/retrieval_config.yaml configs/retrieval_config_local.yaml
# Set your OPENAI_API_KEY (or equivalent) as an environment variable:
export LLM_API_KEY="your-key-here"
```

---

## Running Experiments

```bash
# Run all experiments in sequence
bash scripts/run_all_experiments.sh

# Individual experiments
python scripts/msmarco_ablation.py --config configs/retrieval_config.yaml
python scripts/msdialog_processing.py --config configs/retrieval_config.yaml
python scripts/wixqa_ablation.py --config configs/retrieval_config.yaml
python scripts/hallucination_rate_eval.py --config configs/retrieval_config.yaml --n_queries 500
```

Results are written to `results/`.

---

## How to Cite

If you use this code or build on this work, please cite:

```bibtex
@article{ganesan2026msrag,
  title     = {A Multi-Source Retrieval-Augmented Generation Framework for
               Intelligent Agent Orchestration in Enterprise {CRM} Systems},
  author    = {Ganesan, Chitrapradha},
  journal   = {IEEE Access},
  year      = {2026},
  note      = {DOI to be assigned upon publication}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

This repository is released under the [Apache 2.0 License](LICENSE).
