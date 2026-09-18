# Architecture

## Component flow

```
User (text and/or structured JSON)
  -> Query Understanding / Environmental Variable Extraction  (app/reasoning/variable_extractor.py)
  -> merge with Conversation Memory for this session           (app/conversation/memory.py)
  -> Knowledge Retrieval                                        (app/rag/retriever.py)
  -> Evidence Filtering / Reranking (metadata filter + cosine)  (app/rag/retriever.py)
  -> Multi-Metric Reasoning over the causal graph                (app/reasoning/multi_metric_reasoner.py)
  -> Recommendation Generation                                    (app/reasoning/recommendation_engine.py)
  -> Evidence / Citation Layer (attached inline to each rec)
  -> [optional] LLM phrasing pass, facts frozen                   (app/reasoning/llm_augment.py)
  -> Structured Response                                          (app/models/schemas.py: StructuredResponse)
```

`app/pipeline.py::run_pipeline` is the single place all of this is wired
together; `app/main.py` routes are thin wrappers around it.

## Why this shape, and not something else

### Multi-agent framework vs. a linear pipeline
The brief lists "Specialized Agents/Reasoning Modules" as a *potential*
component, and explicitly says: *"Do not force a multi-agent architecture
if a simpler architecture performs the same task."* This project uses
**named, single-responsibility modules** (extractor, retriever, reasoner,
recommender) that behave like agents conceptually — each has one job, one
input contract, one output contract, independently testable — but they run
as a deterministic in-process pipeline, not as autonomous LLM agents that
negotiate/re-plan. At this hackathon's scope (2–5 known variables per
request, a 12-document knowledge base), an autonomous multi-agent framework
adds latency, cost, and non-determinism without a measurable quality gain.
The trade-off: this project does **not** get automatic quality gains from,
e.g., an evidence-verification agent that catches a bad retrieval — instead
it gets that guarantee statically (the anti-fabrication test suite) and via
metadata filtering at retrieval time. If the knowledge base grows to
hundreds of documents with genuinely conflicting evidence across sources,
promoting "Evidence Verification" from a static guarantee to a runtime
agent step becomes worth the added complexity — noted in
`LIMITATIONS_AND_FUTURE_WORK.md` as a P1/P2 item depending on how far a team
wants to take it.

### Rule-based extraction vs. LLM-based extraction
Regex/keyword extraction (`variable_extractor.py`) is deterministic, has
zero latency/cost, and is directly unit-testable (`tests/test_variable_extractor.py`).
The cost is coverage: a phrasing the regexes don't anticipate (e.g. heavy
slang, non-English input) won't be caught, and the system will ask a
clarifying question instead of extracting silently — which is an acceptable
failure mode (asking is *in spec*; hallucinating a variable value is not).
**Trade-off table:**

| Approach | Accuracy on covered patterns | Coverage of novel phrasing | Latency/cost | Explainability |
|---|---|---|---|---|
| Regex/keyword (chosen) | High | Low-medium | ~0ms, free | Fully explainable |
| LLM-based extraction | Medium-high | High | ~200-800ms, API cost | Requires structured-output validation |
| Hybrid (regex first, LLM fallback only on empty extraction) | High | High | Low avg cost | Explainable in the common case |

The hybrid approach is the recommended next step or Anthropic API model call to make (P1, see `LIMITATIONS_AND_FUTURE_WORK.md`) once regex coverage on real logged queries is known.

### Retrieval: semantic embeddings vs. TF-IDF vs. keyword search
`app/rag/retriever.py` uses `sentence-transformers` (all-MiniLM-L6-v2) +
cosine similarity when installed, and falls back to `scikit-learn`'s TF-IDF
vectorizer (still cosine similarity, just over sparse lexical vectors) when
it isn't. Both paths go through the exact same metadata-filter-then-rank
logic, so retrieval *behavior* is consistent; only recall on paraphrased
queries differs (embeddings generalize past exact wording; TF-IDF does not).
This was chosen specifically so a judge cloning the repo gets a fully working
system with `pip install fastapi ... scikit-learn` alone — no multi-hundred-MB
model download required to evaluate the submission.

A dedicated vector database (Pinecone/Weaviate/Chroma) was considered and
rejected for this scope: the knowledge base is 12 documents. `IndexFlatIP`
via FAISS (or plain NumPy matrix multiply in the fallback) is the exact-search
equivalent of what a vector DB provides for a corpus this size, at zero
infrastructure cost. `docs/RAG_PIPELINE.md` documents exactly how this would
scale to a real ingested-paper corpus (hundreds to thousands of chunks),
where a managed vector DB does start to earn its complexity.

### Knowledge graph
`data/variable_interactions.json` is a small, hand-curated knowledge graph
(12 nodes, 12 edges) rather than a full graph database (Neo4j, etc.). It is
loaded once and traversed in-memory (`multi_metric_reasoner.py`). This was a
deliberate scope decision: a graph DB earns its cost once the graph has
enough nodes/edges that ad-hoc traversal in Python becomes the bottleneck,
or once the graph needs to be edited by non-engineers via a UI. Neither is
true yet at 12 edges. Noted as a P2 differentiator to add if time allows.

## Conversational memory

`app/conversation/memory.py` keeps an in-process dict keyed by `session_id`.
This is intentionally the simplest thing that could work for a hackathon
demo / single-process deployment, and is fully swappable: the
`ConversationMemory` class has get/update/record methods and could be backed
by Redis (for multi-instance horizontal scaling) or Postgres (for durable,
auditable conversation logs) without changing any caller code in
`pipeline.py`. Not implemented here because a single demo process doesn't
need it, and shipping an unused Redis dependency would just be complexity
for its own sake, which the brief explicitly asks us to avoid.

## Data flow for a structured (JSON) request

```
POST /assess {observation: {...}}
  -> AssessRequest validated against EnvironmentalObservation (Pydantic)
  -> wrapped into a ChatRequest with message=""
  -> same run_pipeline() as /chat, so structured and text input share
     100% of the extraction/retrieval/reasoning code path
```

This was a deliberate simplification: rather than maintaining two parallel
reasoning pipelines (one for text, one for JSON), both inputs are normalized
into the same internal `{variable_key: value}` dict as early as possible
(`variable_extractor.py`'s two entry points,
`extract_from_text`/`extract_from_structured`), and everything downstream is
input-mode-agnostic.
