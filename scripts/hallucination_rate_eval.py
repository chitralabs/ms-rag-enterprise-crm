"""
Hallucination rate evaluation script.

Computes hallucination rates across four system configurations:
  1. LLM-only (no retrieval grounding)
  2. BM25-grounded
  3. Dense-grounded
  4. Hybrid MS-RAG

Human annotation labels are loaded from annotations/anonymized_hallucination_labels_sample.csv.
This script computes hallucination rates and inter-annotator agreement (Cohen's kappa).

IMPORTANT: LLM generation results may vary unless the exact model snapshot,
temperature (0.0), and decoding settings are reproduced. See configs/prompt_config.yaml.

Usage:
    python scripts/hallucination_rate_eval.py \
        --annotations annotations/anonymized_hallucination_labels_sample.csv \
        --output results/hallucination_summary.csv
"""

import argparse
import csv
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import hallucination_rate, cohens_kappa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_annotations(csv_path: str) -> list:
    """
    Load human annotation labels from CSV.

    Expected columns:
        query_id, configuration, annotator_a, annotator_b
    where annotator labels are 0 (grounded) or 1 (hallucination).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Annotation file not found: {csv_path}. "
            "See annotations/README.md for the expected format."
        )
    records = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "query_id": row["query_id"],
                "configuration": row["configuration"],
                "annotator_a": int(row["annotator_a"]),
                "annotator_b": int(row["annotator_b"]),
            })
    return records


def compute_metrics(records: list) -> dict:
    """Compute per-configuration hallucination rates and kappa."""
    configs: dict = {}
    for rec in records:
        cfg = rec["configuration"]
        configs.setdefault(cfg, {"a": [], "b": []})
        configs[cfg]["a"].append(rec["annotator_a"])
        configs[cfg]["b"].append(rec["annotator_b"])

    results = {}
    for cfg, labels in configs.items():
        # Majority vote label (agreement = both annotators agree)
        majority = [
            a if a == b else -1  # -1 = disagreement (excluded from rate)
            for a, b in zip(labels["a"], labels["b"])
        ]
        agreed = [l for l in majority if l >= 0]
        rate = hallucination_rate(agreed) if agreed else float("nan")
        kappa = cohens_kappa(labels["a"], labels["b"])
        results[cfg] = {
            "n_queries": len(labels["a"]),
            "n_agreed": len(agreed),
            "hallucination_rate": round(rate, 4),
            "cohens_kappa": round(kappa, 4),
        }
        logger.info(
            "%s — rate: %.3f, kappa: %.3f (n=%d, agreed=%d)",
            cfg, rate, kappa, len(labels["a"]), len(agreed),
        )
    return results


def write_results(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["configuration", "n_queries", "n_agreed", "hallucination_rate", "cohens_kappa"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg, metrics in results.items():
            writer.writerow({"configuration": cfg, **metrics})
    logger.info("Hallucination summary written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Hallucination rate evaluation")
    parser.add_argument(
        "--annotations",
        default="annotations/anonymized_hallucination_labels_sample.csv",
        help="Path to the anonymized annotation CSV",
    )
    parser.add_argument("--output", default="results/hallucination_summary.csv")
    parser.add_argument(
        "--n_queries",
        type=int,
        default=None,
        help="If set, use only the first N annotation records (for subset evaluation)",
    )
    args = parser.parse_args()

    records = load_annotations(args.annotations)
    if args.n_queries:
        records = records[: args.n_queries]

    logger.info("Loaded %d annotation records.", len(records))
    results = compute_metrics(records)
    write_results(results, args.output)


if __name__ == "__main__":
    main()
