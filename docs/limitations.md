# Limitations

This document states the boundaries of the claims made in the paper and the
limitations of the public reproducibility materials in this repository.

---

## Retrieval

**Hybrid retrieval benefit is domain-dependent.**  
On MS MARCO (lexically diverse web-search queries), the hybrid BM25+dense configuration
substantially outperforms dense-only (MRR 0.672 vs 0.641). On WixQA (enterprise KB,
where query and document vocabulary are closely aligned), the hybrid configuration
performs on par with dense-only (MRR 0.501 vs 0.499). The paper states this boundary
explicitly and does not claim universal retrieval superiority for the hybrid approach.

**FAISS index type.**  
Experiments use `IndexFlatIP` (exact search). For very large corpora (multi-million passages),
approximate nearest-neighbor indexes (e.g., IVF, HNSW) would be required; this trades
recall for speed in ways not characterized in the paper.

**BM25 tokenization.**  
The BM25 implementation uses simple whitespace tokenization with punctuation removal.
Language-specific stemmers or sub-word tokenizers may yield different results on
non-English or domain-specific vocabulary.

---

## Evaluation

**MS MARCO passage subset.**  
Ablation scripts default to the first 100,000 passages (`--max_passages 100,000`)
for tractable local experimentation. The full MS MARCO collection contains ~8.8 million
passages; results on the full collection may differ.

**Hallucination rate generalization.**  
The hallucination study uses 500 queries styled after CRM support topics but derived
from public MS MARCO queries. Hallucination rates in real enterprise deployments
will depend on the specific LLM, knowledge base, and query distribution.

**LLM reproducibility.**  
Hallucination rates depend on the LLM model snapshot, API version, and decoding settings.
Results reported use `gpt-4o` at temperature 0.0. Future API deprecations or model
updates may change outputs.

---

## Architecture

**Event-driven synchronization layer.**  
The Kafka-style synchronization design is described at the architecture level.
A full implementation requires live event infrastructure (message broker, consumer
workers) that is not included in this public repository.

**Access control.**  
The access router in `src/orchestration/access_router.py` implements the routing
logic using generic source identifiers. In a real enterprise deployment, source
identifiers map to specific CRM knowledge indices and access-control policies
that are not reproduced here.

**Confidence estimation.**  
The confidence estimator uses a heuristic combination of retrieval score and
query–passage token overlap. A production system may use a calibrated classifier
or LLM self-assessment; the specific calibration is not characterized in the paper.

---

## Scope

This repository provides reproducibility materials for a **public-dataset-based
academic study**. It is not a production-ready CRM system and does not include:
- Real enterprise data or configurations
- Scalability optimizations for production-scale deployments
- End-to-end deployment infrastructure
- Compliance or security review for production use
