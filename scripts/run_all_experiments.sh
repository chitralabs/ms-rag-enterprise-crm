#!/usr/bin/env bash
# run_all_experiments.sh
# Runs all ablation and evaluation scripts in sequence.
# Edit the DATA_DIR variables to match your local dataset paths.
# See data/README.md for dataset download instructions.

set -euo pipefail

CONFIG="configs/retrieval_config.yaml"
MSMARCO_DIR="data/msmarco"
MSDIALOG_DIR="data/msdialog"
WIXQA_DIR="data/wixqa"
RESULTS_DIR="results"
ANNOTATIONS="annotations/anonymized_hallucination_labels_sample.csv"

echo "========================================="
echo " MS-RAG Experiment Suite"
echo "========================================="

echo ""
echo "--- [1/4] MS MARCO ablation ---"
python scripts/msmarco_ablation.py \
    --data_dir "$MSMARCO_DIR" \
    --config "$CONFIG" \
    --top_k 10 \
    --output "$RESULTS_DIR/msmarco_summary_results.csv"

echo ""
echo "--- [2/4] MSDialog processing and evaluation ---"
python scripts/msdialog_processing.py \
    --data_dir "$MSDIALOG_DIR" \
    --config "$CONFIG" \
    --top_k 10 \
    --output "$RESULTS_DIR/msdialog_domain_results.csv"

echo ""
echo "--- [3/4] WixQA ablation ---"
python scripts/wixqa_ablation.py \
    --data_dir "$WIXQA_DIR" \
    --config "$CONFIG" \
    --top_k 10 \
    --output "$RESULTS_DIR/wixqa_ablation_results.csv"

echo ""
echo "--- [4/4] Hallucination rate evaluation ---"
python scripts/hallucination_rate_eval.py \
    --annotations "$ANNOTATIONS" \
    --output "$RESULTS_DIR/hallucination_summary.csv"

echo ""
echo "========================================="
echo " All experiments complete. Results in: $RESULTS_DIR/"
echo "========================================="
