# RAG Pipeline — Design Answers

This answers the nine questions the project brief requires for any proposed
RAG architecture. Code references point to the actual implementation, not a
proposal.

## 1. What is stored
`data/knowledge_base.json`: 12 documents, each representing one scientific
finding/mechanism with its full citation (title, authors, year, source,
DOI/URL where available), a plain-language causal `mechanism`, a
`finding_summary`, a `quantitative_effect` object, an `evidence_type`
(meta-analysis / systematic review / regional field study / government or
IPCC assessment / qualitative consensus), and a `confidence` rating.
`data/variable_interactions.json`: 12 directed edges forming a small causal
graph over 12 canonical environmental variables, each edge pointing back to
the `knowledge_base.json` ids that substantiate it.

## 2. How documents are chunked
**One knowledge base entry = one retrievable chunk.** No further splitting.
At ~120–220 words per entry, each chunk is small enough to embed as a single
unit while staying large enough to keep a finding attached to its mechanism
and its citation in the same chunk — see `app/knowledge/loader.py::documents_as_corpus_text()`.
This was a deliberate choice: naive fixed-length chunking (e.g. 500-token
windows) risks splitting a statistic from the study that produced it, which
is exactly the kind of retrieval bug that causes citation misattribution in
RAG systems. At this corpus size, one-chunk-per-source has no downside.

*(If/when real papers are ingested wholesale — see "Scaling" below — this
project's ingestion script would chunk by section (abstract, methods,
results) with the source's citation duplicated into every chunk's metadata,
specifically to avoid that failure mode at scale.)*

## 3. What metadata is attached
Per document: `id`, `domain[]` (e.g. `soil_health`, `biodiversity`),
`variables_studied[]` (canonical variable keys, matched against extracted
user variables), `region`, `ecosystem`, `evidence_type`, `confidence`,
`year`. This is what makes the retrieval "hybrid" rather than pure
similarity search (see #5).

## 4. How queries are embedded
`app/rag/retriever.py::_embed_query`. Primary path: `sentence-transformers`
`all-MiniLM-L6-v2` (384-dim), L2-normalized. Fallback (zero extra downloads):
`scikit-learn` `TfidfVectorizer` (unigrams+bigrams, English stopwords
removed) fit once over the corpus, L2-normalized. The query is built by
`recommendation_engine.py::generate_recommendations` by joining the user's
known `{variable: value}` pairs into a short natural-language string (e.g.
`"soil_organic_carbon: 0.3; rainfall: low; land_use: monoculture"`) — not by
embedding the user's raw chat message, which tends to contain conversational
filler that dilutes the signal.

## 5. How retrieval works
Cosine similarity between the query vector and every document vector
(`_DOC_MATRIX @ qvec`, both L2-normalized so the dot product *is* cosine
similarity). With FAISS installed this would be `IndexFlatIP`; without it,
the exact same math runs as a NumPy matrix-vector product — identical
ranking, since exact search over 12 documents needs no approximation
structure. **Hybrid step:** before ranking, candidates are filtered to
documents whose `variables_studied[]` overlaps the variable keys extracted
from the user's input — but only if that filter leaves at least one
candidate, so an unusual query never gets zero results just because of an
imperfect variable match (`retrieve()`'s `if filtered:` guard).

## 6. How relevant evidence is selected
After the hybrid filter+rank above, `recommendation_engine.py` de-duplicates
to one recommendation per source (so the same study isn't cited twice under
different wordings) and takes up to the top 3 by score. A `min_score`
floor (0.05) plus a guaranteed-non-empty fallback (`retrieve()`'s trailing
`if not results and scored:` block) balance precision (don't cite a barely-
related source) against never silently returning nothing.

## 7. How evidence is passed to the reasoning model
Full document *dicts* are threaded through the pipeline — not
re-summarized text. `recommendation_engine.py` reads `mechanism`,
`finding_summary`, and `quantitative_effect` directly off the retrieved
dict and copies them, verbatim, into the `Recommendation`/`EvidenceItem`
Pydantic models. There is no step where a language model re-reads and
re-writes the evidence text (unless the optional LLM layer is enabled, and
even then it's a downstream phrasing pass over the *already-final* JSON —
see #9).

## 8. How citations are preserved
Every `Recommendation.evidence` entry carries `kb_id`, `title`, `source`,
`year`, `url`, `evidence_type`, and `confidence` straight from the KB
record — the same object a developer can look up directly via
`GET /knowledge/{kb_id}`. Nothing about the citation is regenerated or
paraphrased at request time.

## 9. How hallucinations are reduced
- **Structural**: the reasoning/recommendation code path is pure Python
  dict/string manipulation over retrieved records — there is no generative
  step that could "fill in" a missing number. If a KB entry's
  `quantitative_effect.value_type` is `qualitative_inference` (i.e. no safe
  number exists), `recommendation_engine.py` explicitly renders that as
  *"Magnitude: not quantified — treat as directional evidence only"*, and
  `tests/test_recommendation_engine.py::test_qualitative_kb_entries_never_produce_a_fabricated_percent_in_expected_impact`
  enforces this in CI.
- **Optional LLM layer**: `llm_augment.py`'s system prompt explicitly
  forbids introducing any new number/citation not already in the JSON it's
  given, and forbids removing existing uncertainty language. If the LLM call
  fails or is disabled, the pipeline silently falls back to a deterministic
  template string (`pipeline.py::_template_assessment_summary`) — the system
  never depends on the LLM being available or well-behaved.
- **Contradictory evidence is preserved, not resolved**: `kb006` (habitat
  fragmentation) is a case where the underlying science is genuinely
  contested; rather than picking a side or averaging into a fake consensus
  number, the entry states the disagreement directly and the recommended
  action (`_ACTION_BY_KB_TOPIC["kb006"]`) is deliberately the
  *lower-risk* lever ("protect total habitat area first") rather than a
  claim about fragmentation *per se*.

## Scaling the retrieval layer (not implemented, documented for reviewers)
If this project ingested a real corpus of dozens/hundreds of full papers
(rather than 12 curated findings) the next steps would be: (a) a proper
ingestion script chunking by section with citation metadata duplicated per
chunk, (b) swapping `IndexFlatIP` for an approximate index (`IndexIVFFlat`
or `IndexHNSWFlat`) once exact search stops being sub-millisecond, (c)
promoting `app/rag/build_index.py` (already included, currently optional)
from a manual script to a CI/CD step that rebuilds and versions the index
artifact on every knowledge base change, and (d) adding a cross-encoder
reranker (e.g. `ms-marco-MiniLM`) as a second-stage reranking step on top of
the current bi-encoder retrieval for higher precision at larger corpus
sizes. None of this is needed at 12 documents; listed here so reviewers can
see the scaling path was considered, not overlooked.
