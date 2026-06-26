"""
MSDialog conversational dataset processing and retrieval evaluation.

MSDialog is a dialogue dataset derived from Microsoft community forums.
This script processes the dataset into query–passage pairs and evaluates
retrieval configurations.

Usage:
    python scripts/msdialog_processing.py \
        --data_dir data/msdialog \
        --config configs/retrieval_config.yaml \
        --output results/msdialog_domain_results.csv

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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_msdialog(data_dir: str) -> tuple:
    """
    Load MSDialog data.

    Expects MSDialog-Intent.json (or the processed train/test split).
    Returns (corpus_passages, queries, qrels) where each passage is a
    candidate response and each query is an utterance.
    """
    json_path = os.path.join(data_dir, "MSDialog-Intent.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"MSDialog data not found at {json_path}. "
            "See data/README.md for download instructions."
        )

    with open(json_path, encoding="utf-8") as f:
        dialogs = json.load(f)

    passages = []
    queries = {}
    qrels = {}
    passage_id = 0

    for dialog_id, dialog in dialogs.items():
        utterances = dialog.get("utterances", [])
        for i, utt in enumerate(utterances[:-1]):
            # Query = current utterance; relevant response = next utterance
            qid = f"{dialog_id}_{i}"
            queries[qid] = utt.get("utterance", "")
            response_text = utterances[i + 1].get("utterance", "")
            pid = str(passage_id)
            passages.append({"id": pid, "text": response_text, "source": "msdialog"})
            qrels[qid] = {pid}
            passage_id += 1

    logger.info("MSDialog: %d queries, %d candidate passages.", len(queries), len(passages))
    return passages, queries, qrels


def run_domain_eval(passages, queries, qrels, cfg, top_k=10):
    """Evaluate BM25, Dense, and Hybrid on MSDialog."""
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
    query_ids = [qid for qid in queries if qid in qrels]

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
    parser = argparse.ArgumentParser(description="MSDialog processing and evaluation")
    parser.add_argument("--data_dir", default="data/msdialog")
    parser.add_argument("--config", default="configs/retrieval_config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output", default="results/msdialog_domain_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    passages, queries, qrels = load_msdialog(args.data_dir)
    results = run_domain_eval(passages, queries, qrels, cfg, top_k=args.top_k)
    write_results(results, args.output)


if __name__ == "__main__":
    main()
