# Reproducibility Checklist

This checklist maps claims in the paper to the code and data needed to reproduce them.

---

## Environment

- [ ] Python 3.9+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] (Optional) Conda environment: `conda env create -f environment.yml`

---

## Datasets

- [ ] MS MARCO collection downloaded to `data/msmarco/` (see `data/README.md`)
- [ ] MS MARCO dev queries and qrels present
- [ ] MSDialog `MSDialog-Intent.json` present in `data/msdialog/`
- [ ] WixQA `articles.jsonl` and `queries.jsonl` present in `data/wixqa/`

---

## MS MARCO Ablation (Table in paper: BM25 / Dense / Hybrid)

- [ ] Run: `python scripts/msmarco_ablation.py --config configs/retrieval_config.yaml`
- [ ] Output: `results/msmarco_summary_results.csv`
- [ ] Expected: Hybrid MRR ≈ 0.672, P@5 ≈ 0.630 (may vary slightly with passage subset)
- [ ] Statistical significance: paired Wilcoxon tests are computed inside the ablation script

---

## MSDialog Evaluation

- [ ] Run: `python scripts/msdialog_processing.py --config configs/retrieval_config.yaml`
- [ ] Output: `results/msdialog_domain_results.csv`

---

## WixQA Ablation (enterprise KB domain)

- [ ] Run: `python scripts/wixqa_ablation.py --config configs/retrieval_config.yaml`
- [ ] Output: `results/wixqa_ablation_results.csv`
- [ ] Expected: Dense MRR ≈ 0.499, Hybrid MRR ≈ 0.501 (on-par, domain-dependent)

---

## Hallucination Rate Study

- [ ] Annotation labels available: `annotations/anonymized_hallucination_labels_sample.csv`
- [ ] Full 500-record annotation set: request from corresponding author
- [ ] Run: `python scripts/hallucination_rate_eval.py`
- [ ] Output: `results/hallucination_summary.csv`
- [ ] Expected: Hybrid rate ≈ 7.8%, LLM-only rate ≈ 34.2%, κ ≈ 0.81

### LLM Generation Reproducibility

> **Note:** LLM generation results (hallucination rates) may vary unless the exact
> model snapshot (`gpt-4o`) and decoding settings (temperature 0.0, max_tokens 512)
> from `configs/prompt_config.yaml` are used. The hallucination rate reported in the
> paper is based on human annotation of LLM outputs generated with these settings.

- [ ] `LLM_API_KEY` environment variable set
- [ ] Model: `gpt-4o` as specified in `configs/prompt_config.yaml`
- [ ] Temperature: 0.0 (deterministic)

---

## Run All

- [ ] `bash scripts/run_all_experiments.sh`

---

## Known Sources of Variation

| Source | Impact | Mitigation |
|--------|--------|-----------|
| MS MARCO passage subset size | MRR/P@5 may differ from full-corpus evaluation | Set `--max_passages` to a larger value |
| Dense model version | Embeddings differ across model checkpoints | Use `all-MiniLM-L6-v2` from HuggingFace Hub |
| LLM model snapshot | Output tokens differ across API versions | Use `gpt-4o` snapshot; note any deprecations |
| Random FAISS ordering | Tie-breaking may differ | IndexFlatIP is deterministic for same embeddings |
