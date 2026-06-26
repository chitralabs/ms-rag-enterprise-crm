# Architecture Overview: MS-RAG Framework

## Five-Layer Design

The MS-RAG framework is organized into five vertical layers, each with a distinct
responsibility. The design is motivated by deployment constraints in enterprise CRM
partner portals; the evaluation uses public datasets.

---

### Layer 1: Multi-Source Ingestion

**Purpose:** Normalize heterogeneous knowledge sources into a unified passage representation.

**Source types addressed:**
| Source Type | Characteristics |
|-------------|----------------|
| Structured knowledge articles | Versioned, schema-consistent, frequently updated |
| CMS repositories | Semi-structured, rich metadata, varied update cadence |
| PDF documents | Unstructured, layout-dependent, static |
| Cloud-hosted files | Variable format, access-controlled |

**Processing steps:**
1. Source-specific parsers extract raw text and metadata
2. Schema normalization maps heterogeneous fields to a common document schema: `{id, text, source, updated_at, access_level, ...}`
3. Chunking splits long documents into overlapping passages (chunk size and overlap are configurable in `configs/retrieval_config.yaml`)
4. Metadata tagging attaches source identifier and access-level label to each passage

---

### Layer 2: Hybrid Retrieval

**Purpose:** Retrieve a diverse candidate set combining lexical and semantic signals.

**Components:**

#### BM25 (Sparse)
- Indexes tokenized passage text using `rank-bm25` (BM25Okapi variant)
- Efficient for exact keyword matching and out-of-vocabulary terms
- Parameters: k1 = 1.5, b = 0.75 (standard defaults; see `configs/retrieval_config.yaml`)

#### Bi-Encoder Dense Retrieval (FAISS)
- Encodes passages and queries using a sentence-transformer model (`all-MiniLM-L6-v2`)
- Stores passage embeddings in a FAISS flat inner-product index (exact search)
- Effective for semantic paraphrase and synonym matching

#### Reciprocal Rank Fusion (RRF)
- Combines ranked lists from BM25 and dense retrieval
- RRF score: Σ_r 1 / (k + rank_r(d)), k = 60
- Does not require score normalization across retrievers
- Reference: Cormack, Clarke & Buettcher, SIGIR 2009

#### SHA-256 Deduplication
- Removes exact-duplicate passages using content hashing before LLM input
- Prevents identical content from occupying multiple context slots

---

### Layer 3: LLM Grounding and Generation

**Purpose:** Generate factually grounded responses from retrieved passages.

**Prompt structure:**
- System prompt: constrains the LLM to answer only from provided passages
- User prompt: query + numbered context passages with source attribution
- Explicit fallback instruction: models must state when passages are insufficient

**Hallucination mitigation mechanism:** the grounding constraint in the system prompt
reduces hallucination from 34.2% (LLM-only) to 7.8% (hybrid-grounded) across
500 annotated CRM-style queries.

---

### Layer 4: Dual-Agent Orchestration

**Purpose:** Route queries to appropriate agent configurations based on user access level.

**Access classes:**

| Class | Sources Accessible | Prompt Template | Confidence Threshold |
|-------|-------------------|-----------------|---------------------|
| Authenticated | All four source types | `authenticated_agent` | 0.60 |
| Public | Public knowledge articles + CMS | `public_agent` | 0.65 |

**Confidence-based escalation:**
- Confidence estimated from top-passage retrieval score and query–passage token overlap
- Queries below the threshold are routed to a human support queue
- Prevents low-confidence responses from being returned to users

---

### Layer 5: Event-Driven Synchronization

**Purpose:** Keep BM25 and FAISS indexes current as connected data sources update.

**Design (Kafka-style):**
- Source systems publish change events to a topic/queue
- A synchronization consumer processes events and updates the relevant index partitions
- Source-specific sync modes:
  - **Push** (webhook): knowledge article systems push update events
  - **Poll**: CMS repositories are polled on a configurable schedule
  - **Batch**: PDF and cloud file ingestion runs on a nightly schedule

This layer is described at the design level in the paper and is not fully implemented
in this public repository (it requires live event infrastructure).

---

## Data Flow Summary

```
Query
  │
  ▼
AccessRouter ──────────────► AgentConfig (sources, template, threshold)
  │
  ▼
BM25Retriever ─┐
               ├─► RRF Fusion ─► Deduplication ─► Top-m Passages
DenseRetriever ┘
  │
  ▼
PromptBuilder (system + user prompt with grounded context)
  │
  ▼
LLM API call ─► Generated Response
  │
  ▼
ConfidenceEscalator ──► Return to user  OR  Escalate to human support
```
