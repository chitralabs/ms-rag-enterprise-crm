# Hallucination Annotation Guidelines

**Study:** MS-RAG Hallucination Rate Evaluation  
**Task:** Binary annotation of LLM-generated responses to CRM-style queries  
**Label set:** 0 = Factually Grounded, 1 = Hallucination Present

---

## Annotator Task

Given:
- A **query** (a question in the style of an enterprise CRM support request)
- A set of **retrieved context passages** (the passages provided to the LLM)
- A **generated response** (the LLM's answer)

Assign a binary label:
- **0 (Grounded):** Every factual claim in the response can be traced to at least one of the provided context passages.
- **1 (Hallucination):** The response contains one or more factual claims that cannot be verified from the provided passages, OR that directly contradict the passages.

---

## Definition of Hallucination

A response is labelled as hallucinated (**label = 1**) if it contains any of:

1. **Fabricated facts:** Specific claims (version numbers, pricing, steps, deadlines, product names, features) not present in the context passages.
2. **Source contradictions:** Claims that directly contradict information in the passages.
3. **Unsupported specificity:** General statements converted to specific numbers or procedures not in the context.
4. **Invented citations:** References to documents, policies, or sources not mentioned in the passages.

A response is **not** hallucinated if:
- It appropriately declines to answer ("I don't have enough information in the available knowledge base…")
- It paraphrases passage content faithfully without adding new facts
- It uses connective language (e.g., "therefore," "as a result") to logically link passage content

---

## Edge Cases

| Situation | Label |
|-----------|-------|
| Response says "I don't know" or refuses to answer | **0** (grounded; correct behavior) |
| Response contains one unsupported claim amid otherwise grounded content | **1** |
| Minor wording variation (synonym, paraphrase) that preserves meaning | **0** |
| Response adds commonsense inference not in passages | **1** (unsupported specificity) |
| Passage is ambiguous; response picks one interpretation | **0** (if interpretation is reasonable) |

---

## Annotation Procedure

1. Read the query carefully.
2. Read all context passages.
3. Read the generated response.
4. For each factual claim in the response, check whether it is supported by at least one passage.
5. Assign label 0 if all claims are supported; assign label 1 if any claim is not.
6. Do not consider world knowledge outside the passages when assigning labels.

---

## Quality Control

- Both annotators independently label each response before discussing.
- Disagreements are resolved by a third review; if unresolved, the record is excluded.
- Inter-annotator agreement is assessed using Cohen's κ; κ ≥ 0.80 is the target threshold.

---

## Ethical Notes

- Query texts are derived from publicly available MS MARCO topics, not from real customer interactions.
- No real customer data, employer data, or proprietary CRM content appears in this annotation set.
