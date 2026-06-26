"""
MS MARCO Passage Ranking ablation script.

Compares BM25-only, Dense-only, and Hybrid (MS-RAG) retrieval configurations
on a subset of the MS MARCO Passage Ranking dataset.

Usage:
    python scripts/msmarco_ablation.py \
        --data_dir data/msmarco \
        --config configs/retrieval_config.yaml \
        --top_k 10 \
        --output results/msmarco_summary_results.csv

Dataset download: see data/README.md
"""

import argparse
import csv
import json
import logging
import os
import sys

import yaml
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf_fusion import reciprocal_rank_fusion
from src.retrieval.deduplication import deduplicate
from src.evaluation.metrics import mean_reciprocal_rank, mean_precision_at_k, mean_rouge_l

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_msmarco_passages(data_dir: str, max_passages: int = 100_000) -> list:
    """Load MS MARCO passage collection (collection.tsv)."""
    passages = []
    tsv_path = os.path.join(data_dir, "collection.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"MS MARCO collection not found at {tsv_path}. "
            "See data/README.md for download instructions."
        )
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= max_passages:
                break
            if len(row) >= 2:
                passages.append({"id": row[0], "text": row[1], "source": "msmarco"})
    logger.info("Loaded %d passages from MS MARCO collection.", len(passages))
    return passages


def load_msmarco_queries_and_qrels(data_dir: str, split: str = "dev") -> tuple:
    """Load queries and qrels for the given split."""
    queries = {}
    queries_path = os.path.join(data_dir, f"queries.{split}.tsv")
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    with open(queries_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                queries[row[0]] = row[1]

    qrels = {}  # qid -> set of relevant pids
    qrels_path = os.path.join(data_dir, f"qrels.{split}.tsv")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"Qrels file not found: {qrels_path}")
    with open(qrels_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 4 and int(row[3]) > 0:
                qrels.setdefault(row[0], set()).add(row[2])

    return queries, qrels


def run_ablation(passages, queries, qrels, cfg, top_k=10):
    """Build indexes and evaluate three retrieval configurations."""
    logger.info("Building BM25 index ...")
    bm25 = BM25Retriever(passages)

    logger.info("Building dense index ...")
    dense = DenseRetriever(passages, model_name=cfg.get("dense_model", "all-MiniLM-L6-v2"))

    configs_to_eval = {
        "bm25_only": lambda q: bm25.retrieve(q, top_k=top_k),
        "dense_only": lambda q: dense.retrieve(q, top_k=top_k),
        "hybrid_rrf": lambda q: reciprocal_rank_fusion(
            [bm25.retrieve(q, top_k=top_k), dense.retrieve(q, top_k=top_k)],
            k=cfg.get("rrf_k", 60),
            top_m=top_k,
        ),
    }

    results = {}
    query_ids = [qid for qid in queries if qid in qrels]
    logger.info("Evaluating on %d queries with qrels.", len(query_ids))

    for config_name, retrieve_fn in configs_to_eval.items():
        logger.info("Evaluating configuration: %s", config_name)
        all_ranked = []
        all_relevant = []

        for qid in tqdm(query_ids, desc=config_name):
            query_text = queries[qid]
            retrieved = retrieve_fn(query_text)
            ranked_ids = [r["id"] for r in retrieved]
            all_ranked.append(ranked_ids)
            all_relevant.append(list(qrels[qid]))

        mrr = mean_reciprocal_rank(all_ranked, all_relevant)
        p5 = mean_precision_at_k(all_ranked, all_relevant, k=5)
        results[config_name] = {"MRR": mrr, "P@5": p5}
        logger.info("%s — MRR: %.4f, P@5: %.4f", config_name, mrr, p5)

    return results


def write_results(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["configuration", "MRR", "P@5"])
        writer.writeheader()
        for config_name, metrics in results.items():
            writer.writerow({"configuration": config_name, **metrics})
    logger.info("Results written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="MS MARCO retrieval ablation")
    parser.add_argument("--data_dir", default="data/msmarco", help="Path to MS MARCO data directory")
    parser.add_argument("--config", default="configs/retrieval_config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_passages", type=int, default=100_000)
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--output", default="results/msmarco_summary_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    passages = load_msmarco_passages(args.data_dir, max_passages=args.max_passages)
    queries, qrels = load_msmarco_queries_and_qrels(args.data_dir, split=args.split)
    results = run_ablation(passages, queries, qrels, cfg, top_k=args.top_k)
    write_results(results, args.output)


if __name__ == "__main__":
    main()
