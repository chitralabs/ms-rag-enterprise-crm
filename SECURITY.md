# Security Policy

## Scope

This repository contains academic research code for a public-dataset-based study.
It is not a production system. The following security guidance applies to anyone
building on or extending this code.

## API Key Handling

**Never hard-code API keys in source code or configuration files.**

All LLM API keys must be provided via environment variables:

```bash
export LLM_API_KEY="your-key-here"
```

The code reads keys using `os.environ.get("LLM_API_KEY")` and raises a clear error
if the variable is not set. Configuration files include only the environment variable
name, not the value (see `configs/prompt_config.yaml`).

The `.gitignore` in this repository excludes `.env` files and any file matching
`*_key.txt` or `credentials.json`.

## Reporting Vulnerabilities

If you discover a security vulnerability in this repository, please report it
by emailing **chitracrmexpert@gmail.com** rather than opening a public GitHub issue.

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 7 business days.

## Data Safety

- This repository does not process real customer data.
- Do not use this code with real PII or confidential enterprise data without
  implementing appropriate access controls, encryption, and compliance review.
- Datasets downloaded for evaluation (MS MARCO, MSDialog, WixQA) are subject
  to their own terms of use; do not redistribute them.

## Dependency Security

Dependencies are pinned to minimum versions in `requirements.txt`.
Run `pip audit` or `safety check` periodically to detect known vulnerabilities
in dependencies:

```bash
pip install safety
safety check -r requirements.txt
```
