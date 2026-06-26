# Results Directory

This directory contains summary outputs from the ablation studies reported in the paper.
All results are based on public datasets only; no proprietary data, internal metrics,
or commercial deployment data is included.

## Files

| File | Description |
|------|-------------|
| `msmarco_summary_results.csv` | BM25 / Dense / Hybrid MRR and P@5 on MS MARCO dev set |
| `msdialog_domain_results.csv` | MRR and P@5 on MSDialog conversational QA |
| `wixqa_ablation_results.csv` | MRR and P@5 on WixQA enterprise KB |
| `hallucination_summary.csv` | Per-configuration hallucination rates and inter-annotator kappa |

## Reproducing

Run `bash scripts/run_all_experiments.sh` after downloading the datasets
(see `data/README.md`) to regenerate all result files.

## Notes on LLM Reproducibility

Results involving LLM-generated answers (hallucination rate experiments) may
vary unless the exact model snapshot (`gpt-4o`) and decoding settings
(temperature 0.0, max_tokens 512) specified in `configs/prompt_config.yaml` are used.
