# Database / Schema Documentation

This project's "database" for the hackathon submission is two structured
JSON files under `/data`, treated as the source of truth and loaded into an
in-memory index at process start (`app/knowledge/loader.py`,
`app/rag/retriever.py`). This section documents them as if they were tables,
because that's the right mental model for anyone extending this into a real
Postgres/vector-DB deployment.

## Table: `knowledge_base` (`data/knowledge_base.json` → `documents[]`)

| Field | Type | Notes |
|---|---|---|
| `id` | string, PK | e.g. `kb001`. Referenced by `variable_interactions.edges[].supporting_kb[]` |
| `title` | string | Paper/report title |
| `authors` | string | Author list or organization |
| `year` | int | Publication year |
| `source` | string | Journal / publisher / report name |
| `doi` | string \| null | DOI where one exists |
| `url` | string \| null | Direct link to the source |
| `evidence_type` | string (enum-ish) | `meta-analysis`, `systematic review`, `regional field study`, `government/policy report`, `qualitative/established consensus`, etc. |
| `domain[]` | string[] | e.g. `soil_health`, `biodiversity`, `climate`, `water`, `human_impact` |
| `variables_studied[]` | string[] | Canonical variable keys (foreign key into `variable_interactions.variables[].key`) |
| `region` | string | Geographic scope of the underlying study |
| `ecosystem` | string | e.g. `cropland`, `orchard`, `riparian`, `drylands` |
| `mechanism` | text | Plain-language causal explanation |
| `finding_summary` | text | The actual result, in the system's own words (not a verbatim quote from the source) |
| `quantitative_effect` | object `{metric, direction, magnitude, value_type}` | `value_type` ∈ `published_measurement`, `model_derived_estimate`, `qualitative_inference` — see anti-fabrication note below |
| `evidence_notes` | text | Caveats, variability across studies, how to use this entry responsibly |
| `confidence` | string | `high` / `medium` / `low`, or a qualified string for mixed cases |

**Example record** (`kb007`, abbreviated):
```json
{
  "id": "kb007",
  "title": "Riparian Buffer Width, Vegetative Cover, and Nitrogen Removal Effectiveness",
  "year": 2016,
  "source": "NC DEQ riparian buffer science review",
  "url": "https://www.deq.nc.gov/.../nccnbufferlitreview2016/download",
  "evidence_type": "government literature synthesis of peer-reviewed field studies",
  "variables_studied": ["water_availability", "pollution", "habitat_diversity"],
  "quantitative_effect": {
    "metric": "pollution", "direction": "decrease",
    "magnitude": "26ft:-28% N, 49ft:-48% N, 100-165ft: up to -85% N",
    "value_type": "published_measurement"
  },
  "confidence": "high"
}
```

**Indexing strategy**: `id` is the primary lookup key (`GET /knowledge/{id}`,
O(1) via a linear scan at this corpus size — trivial to replace with a dict
keyed by id if the corpus grows). For retrieval, the *real* index is the
embedding matrix built once at startup over `documents_as_corpus_text()`
(see `RAG_PIPELINE.md` #4–5); `variables_studied[]` acts as a secondary,
pre-similarity metadata filter (a poor-man's inverted index — trivial to
back with a real Postgres GIN index on a jsonb/array column at scale).

**Retrieval strategy**: hybrid — metadata filter on `variables_studied[]`
(when it doesn't empty the candidate set) followed by cosine-similarity
ranking on the embedding index. See `RAG_PIPELINE.md`.

## Table: `variable_interactions` (`data/variable_interactions.json`)

### `variables[]`
| Field | Type | Notes |
|---|---|---|
| `key` | string, PK | Canonical variable identifier, e.g. `soil_organic_carbon` |
| `label` | string | Human-readable name |
| `unit` | string \| null | Unit of measurement, or `categorical`/`qualitative` |
| `category` | string | `soil`, `land`, `biodiversity`, `climate`, `water`, `human_impact` |

### `edges[]`
| Field | Type | Notes |
|---|---|---|
| `id` | string, PK | e.g. `e1` |
| `from` | string, FK → `variables[].key` | |
| `to` | string, FK → `variables[].key` | |
| `relationship` | text | Plain-language mechanism connecting `from` → `to` |
| `supporting_kb[]` | string[], FK → `knowledge_base.documents[].id` | Which sources substantiate this specific causal claim |

**Example record** (`e3`):
```json
{
  "id": "e3", "from": "soil_moisture", "to": "species_richness",
  "relationship": "Soil moisture constrains which plant and soil-faunal species can survive at a site; chronic moisture stress filters out moisture-sensitive species, narrowing richness.",
  "supporting_kb": ["kb008", "kb010"]
}
```

**Relationships**: this is, structurally, a directed graph / adjacency
list. `app/reasoning/multi_metric_reasoner.py::build_chains` performs a
depth-limited DFS from every variable key the user has provided, so a
request with `{rainfall, soil_organic_carbon, land_use}` surfaces multi-hop
chains like `land_use -> habitat_diversity -> species_richness` even though
`habitat_diversity` and `species_richness` were never directly provided by
the user.

**Integrity checks**: `.github/workflows/ci.yml` validates on every push
that (a) no duplicate `knowledge_base` ids exist, (b) every document has its
required fields populated, and (c) every edge's `supporting_kb[]` references
an id that actually exists in `knowledge_base.json` — i.e. the graph can
never cite a source that isn't really there.

## Conversation state (not persisted — see `app/conversation/memory.py`)
A third, ephemeral "table" exists only in-process: `SessionState` keyed by
`session_id`, holding `known_variables`, `turns[]` (chat history), and
`asked_questions[]` (so the same clarifying question is never asked twice in
one session). Documented here for completeness; see `ARCHITECTURE.md` for
why this isn't backed by a real database at this project's current scope,
and what swapping in Redis/Postgres would look like.

## Migration path to a real database (P2, not implemented)
- `knowledge_base.json` → Postgres table `kb_documents` with a `jsonb`
  column for `quantitative_effect`, a `text[]` column for
  `variables_studied`/`domain`, and a `pgvector` column for the embedding —
  at which point `retriever.py`'s in-memory matrix multiply becomes a single
  `ORDER BY embedding <=> query_embedding LIMIT k` SQL query.
- `variable_interactions.json` → either two Postgres tables
  (`env_variables`, `env_variable_edges`) or, if the graph grows
  significantly, a proper graph database (Neo4j) so multi-hop traversal
  doesn't need to be hand-rolled DFS in application code.
- `SessionState` → Redis (hot session data, TTL-expired) or Postgres
  (durable, auditable conversation logs) depending on whether conversation
  history needs to survive a restart / be queried later.
