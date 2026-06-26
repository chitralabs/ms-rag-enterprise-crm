# Annotations Directory

This directory contains hallucination annotation guidelines and anonymized sample labels
from the human annotation study reported in the paper.

## Study Overview

- **Task:** Binary hallucination annotation of LLM-generated CRM-style query responses
- **Configurations annotated:** LLM-only, BM25-grounded, Dense-grounded, Hybrid MS-RAG
- **Total queries annotated:** 500 CRM-style queries (drawn from public MS MARCO topics)
- **Annotators:** 2 independent annotators
- **Inter-annotator agreement:** Cohen's κ = 0.81

## Files

| File | Description |
|------|-------------|
| `hallucination_annotation_guidelines.md` | Full annotation guidelines provided to annotators |
| `anonymized_hallucination_labels_sample.csv` | Sample of 50 anonymized annotation records |

## Notes

- Query texts are derived from publicly available MS MARCO queries.
- No customer data, employer data, or proprietary CRM content appears in these records.
- Annotator identities are not disclosed; labels are referred to as annotator_a and annotator_b.
- The full annotation set is available from the corresponding author upon reasonable request,
  subject to the MS MARCO license terms.
