# Multi-Source RAG Framework for Intelligent Agent Orchestration in Enterprise CRM Systems

**Manuscript:** "A Multi-Source Retrieval-Augmented Generation Framework for Intelligent Agent Orchestration in Enterprise CRM Systems"  
**Target venue:** IEEE Access (manuscript under preparation)  
**Author:** Chitrapradha Ganesan (Senior Member, IEEE) — The University of Texas at Austin  
**Contact:** chitracrmexpert@gmail.com | ORCID: 0009-0009-1305-1724

---

> **Disclaimer:** This repository contains reproducibility materials for a public-dataset-based academic study. It does not contain proprietary data, employer data, customer records, confidential CRM configurations, internal system information, or commercial deployment metrics. All experiments are designed around publicly available datasets.

---

## Reviewer Note

This repository is provided as supplementary reproducibility material for manuscript review. The repository includes scripts, configuration files, summary outputs, and annotation templates aligned with the manuscript. Full public datasets are not redistributed and should be downloaded from their official sources.

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

### MS MARCO Passage Ranking — Retrieval Ablation

| Configuration | ROUGE-L | MRR | P@5 | Latency (ms) |
|--------------|---------|-----|-----|-------------|
| BM25 Keyword-Only | 0.312 | 0.521 | 0.460 | 187 |
| Dense Vector-Only | 0.371 | 0.583 | 0.540 | 312 |
| **MS-RAG Proposed** | **0.429** | **0.672** | **0.630** | **274** |

All differences p < 0.001, paired Wilcoxon test.

### WixQA Enterprise KB — Retrieval Ablation (n = 200 queries)

| Configuration | MRR | P@1 | P@5 | Recall@m |
|--------------|-----|-----|-----|---------|
| BM25 Keyword-Only | 0.340 | 0.205 | 0.113 | 0.464 |
| Dense Vector-Only | 0.499 | 0.335 | 0.152 | 0.634 |
| **MS-RAG Proposed** | **0.501** | **0.365** | **0.152** | 0.619 |

Note: on WixQA (closely aligned query/document vocabulary), the hybrid configuration performs on par with dense-only rather than substantially above it — the benefit of the sparse BM25 signal is domain-dependent.

### Hallucination Rate (500 CRM-style queries, human-annotated)

| Configuration | Hallucination Rate | Relative Reduction vs. LLM-Only |
|--------------|-------------------|--------------------------------|
| LLM-Only (no retrieval) | 34.2% | — |
| BM25 Keyword-Only | 18.7% | −45.3% |
| Dense Vector-Only | 14.3% | −58.2% |
| **MS-RAG Proposed** | **7.8%** | **−77.2%** |

Inter-annotator agreement: κ = 0.81.

> **Note:** LLM generation results may vary unless the exact model snapshot, temperature, and decoding settings are matched. See [`configs/prompt_config.yaml`](configs/prompt_config.yaml) for the settings used in the manuscript.

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

```bash
cp configs/retrieval_config.yaml configs/retrieval_config_local.yaml
# Set your LLM API key as an environment variable — never hard-code credentials:
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
python scripts/hallucination_rate_eval.py --annotations annotations/anonymized_hallucination_labels_sample.csv
```

Results are written to `results/`.

---

## How to Cite

If you use this code, please cite the manuscript and repository:

```bibtex
@misc{ganesan2026msrag,
  title     = {A Multi-Source Retrieval-Augmented Generation Framework for
               Intelligent Agent Orchestration in Enterprise {CRM} Systems},
  author    = {Ganesan, Chitrapradha},
  year      = {2026},
  note      = {Manuscript prepared for IEEE Access. Repository: https://github.com/chitralabs/ms-rag-enterprise-crm}
}
```

See also [`CITATION.cff`](CITATION.cff).

---

## License

This repository is released under the [Apache 2.0 License](LICENSE).
