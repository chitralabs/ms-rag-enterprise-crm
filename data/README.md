# Data Directory

> **Disclaimer:** This repository contains reproducibility materials for a public-dataset-based academic study. It does not contain proprietary data, employer data, customer records, confidential CRM configurations, internal system information, or commercial deployment metrics. All experiments are designed around publicly available datasets.

This directory contains **no dataset files**. Full datasets are not redistributed
in this repository in compliance with each dataset's terms of use.

All experiments use publicly available datasets. Download instructions for each
dataset are provided below.

---

## Datasets Used in the Paper

### 1. MS MARCO Passage Ranking

**Description:** Large-scale passage retrieval benchmark derived from Bing web queries.  
**Official site:** https://microsoft.github.io/msmarco/  
**License:** MS MARCO License (non-commercial research use)

**Download:**
```bash
mkdir -p data/msmarco && cd data/msmarco

# Passage collection (~3 GB compressed)
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
tar -xzf collection.tar.gz

# Dev queries
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
tar -xzf queries.tar.gz

# Dev qrels
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
```

Expected files in `data/msmarco/`:
- `collection.tsv` — passage ID + text, tab-separated
- `queries.dev.tsv` — query ID + text
- `qrels.dev.tsv` — relevance judgements (query_id, 0, passage_id, relevance)

For manageable experimentation, the ablation scripts default to the first 100,000 passages
(`--max_passages 100000`). Set a higher value or remove the limit for full-scale evaluation.

---

### 2. MSDialog

**Description:** Conversational question answering dataset from Microsoft community forums.  
**Official site:** https://ciir.cs.umass.edu/downloads/msdialog/  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**Download:**
```bash
mkdir -p data/msdialog && cd data/msdialog
# Follow the download link on the official page and place:
# MSDialog-Intent.json  in data/msdialog/
```

---

### 3. WixQA

**Description:** Enterprise customer-support knowledge base with article-level relevance labels.  
**Official repository:** https://github.com/wix-incubator/WixQA  
**License:** MIT License

**Download:**
```bash
git clone https://github.com/wix-incubator/WixQA.git data/wixqa_repo
# Copy the relevant data files:
cp data/wixqa_repo/data/articles.jsonl data/wixqa/
cp data/wixqa_repo/data/queries.jsonl data/wixqa/
```

Expected files in `data/wixqa/`:
- `articles.jsonl` — KB articles (id, title, body)
- `queries.jsonl` — queries with relevant article IDs

---

## Notes

- Do not commit dataset files to this repository.
- Dataset download links may change; check official sites if links are broken.
- All datasets listed above are publicly available for non-commercial research.
- See `configs/dataset_config.yaml` for the expected local paths used by the scripts.
