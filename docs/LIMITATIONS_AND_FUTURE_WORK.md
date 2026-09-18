# Limitations & Future Work

## Priority key
- **P0** — required for a working submission (all implemented)
- **P1** — high-value differentiator, feasible to add with more hackathon time
- **P2** — optional / production-scale enhancement, out of scope for this submission

## P0 — implemented
- Structured knowledge base with real citations (12 entries)
- Hybrid retrieval (embeddings/TF-IDF + metadata filtering)
- Multi-metric reasoning over an explicit causal graph
- Evidence-backed, structured recommendations (what/why/mechanism/metrics/impact/horizon/evidence/confidence)
- Conversational memory + clarifying questions, no repeated questions
- Text input (mandatory) + structured JSON input (bonus) via `/chat` and `/assess`
- Anti-fabrication guardrails, enforced by tests, not just documentation
- CI pipeline validating both code and knowledge base integrity

## P1 — not implemented here, clearly feasible next
- **Hybrid extraction fallback**: route to an LLM-based extractor only when
  regex extraction finds nothing, keeping the deterministic path as default.
- **Geo-spatial context** (bonus requirement): `GeoContext` is already in the
  schema (`lat/lon/region_name`) but isn't yet used to bias retrieval toward
  region-matched KB entries (e.g. prefer `kb005`'s Brazilian semi-arid study
  for a query with `region_name` containing "semi-arid").
- **Confidence scoring calibration**: replace the current rule-based
  confidence mapping with a proper scoring function that also accounts for
  how many independent sources agree vs. how much retrieval score margin
  exists between the top result and the runner-up.
- **What-if simulation**: given a proposed intervention, walk the causal
  graph forward to estimate second-order effects (e.g. "if SOC rises from
  0.3% to 0.5%, which downstream variables should also be expected to
  shift, and by how much confidence").

## P2 — production-scale, explicitly out of scope for this submission
- Persisted vector DB (pgvector/Pinecone) once the corpus exceeds a few
  hundred documents — see `docs/RAG_PIPELINE.md` "Scaling the retrieval layer"
- Redis/Postgres-backed conversation memory for multi-instance deployments
- A knowledge graph database (Neo4j) if the variable-interaction graph grows
  large enough that in-memory DFS traversal becomes a bottleneck or needs a
  non-engineer-editable UI
- Authentication, rate limiting, and per-tenant API keys
- Automated literature ingestion pipeline (PDF -> chunked, cited KB entries)
  to grow the knowledge base beyond hand-curated entries
- Cross-encoder reranking stage for higher precision at larger corpus sizes

## Why these were deprioritized, specifically
Per the brief's own guidance ("Do not add features simply because they
sound impressive... Prioritize a reliable end-to-end pipeline before adding
advanced features"), this submission optimizes for a **fully working,
tested, non-fabricating pipeline** over a longer feature list with weaker
guarantees. Every P1/P2 item above was evaluated against: why it matters,
how it would work, implementation complexity, and whether it's essential or
optional — and none change the answer to "does this system behave like an
AI environmental scientist rather than a chatbot," which the P0 list above
already establishes.
