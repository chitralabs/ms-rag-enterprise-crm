"""
WixQA enterprise KB ablation script.

WixQA is a genuine enterprise customer-support knowledge base with
article-level relevance labels. This script evaluates BM25, Dense,
and Hybrid retrieval configurations on the WixQA benchmark.

Key finding from the paper: the benefit of adding the sparse BM25 signal
is domain-dependent. On WixQA (closely aligned query/document vocabulary),
the hybrid configuration performs on par with dense-only (MRR 0.501 vs 0.499)
rather than substantially above it, unlike on lexically diverse MS MARCO.

Usage:
    python scripts/wixqa_ablation.py \
        --data_dir data/wixqa \
        --config configs/retrieval_config.yaml \
        --output results/wixqa_ablation_results.csv

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf_fusion import reciprocal_rank_fusion
from src.evaluation.metrics import mean_reciprocal_rank, mean_precision_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_wixqa(data_dir: str) -> tuple:
    """
    Load WixQA dataset.

    Expects the following files (as distributed in the official WixQA repo):
      - articles.jsonl  (KB articles: id, title, body)
      - queries.jsonl   (queries: id, text, relevant_article_ids)
    """
    articles_path = os.path.join(data_dir, "articles.jsonl")
    queries_path = os.path.join(data_dir, "queries.jsonl")

    for p in (articles_path, queries_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"WixQA file not found: {p}. "
                "See data/README.md for download instructions."
            )

    passages = []
    with open(articles_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("title", "") + " " + obj.get("body", "")
            passages.append({"id": str(obj["id"]), "text": text.strip(), "source": "wixqa"})

    queries = {}
    qrels = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["id"])
            queries[qid] = obj["text"]
            qrels[qid] = set(str(a) for a in obj.get("relevant_article_ids", []))

    logger.info("WixQA: %d articles, %d queries.", len(passages), len(queries))
    return passages, queries, qrels


def run_ablation(passages, queries, qrels, cfg, top_k=10):
    logger.info("Building BM25 index ...")
    bm25 = BM25Retriever(passages)

    logger.info("Building dense index ...")
    dense = DenseRetriever(passages, model_name=cfg.get("dense_model", "all-MiniLM-L6-v2"))

    configs = {
        "bm25_only": lambda q: bm25.retrieve(q, top_k=top_k),
        "dense_only": lambda q: dense.retrieve(q, top_k=top_k),
        "hybrid_rrf": lambda q: reciprocal_rank_fusion(
            [bm25.retrieve(q, top_k=top_k), dense.retrieve(q, top_k=top_k)],
            k=cfg.get("rrf_k", 60),
            top_m=top_k,
        ),
    }

    results = {}
    query_ids = [qid for qid in queries if qrels.get(qid)]

    for config_name, retrieve_fn in configs.items():
        logger.info("Evaluating %s ...", config_name)
        all_ranked, all_relevant = [], []
        for qid in tqdm(query_ids, desc=config_name):
            retrieved = retrieve_fn(queries[qid])
            all_ranked.append([r["id"] for r in retrieved])
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
        for name, metrics in results.items():
            writer.writerow({"configuration": name, **metrics})
    logger.info("Results written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="WixQA enterprise KB ablation")
    parser.add_argument("--data_dir", default="data/wixqa")
    parser.add_argument("--config", default="configs/retrieval_config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="results/wixqa_ablation_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    passages, queries, qrels = load_wixqa(args.data_dir)
    results = run_ablation(passages, queries, qrels, cfg, top_k=args.top_k)
    write_results(results, args.output)


if __name__ == "__main__":
    main()
