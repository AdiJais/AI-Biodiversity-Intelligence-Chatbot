# Demo Walkthrough (Phase 9)

A ~3 minute path through the system that touches every evaluation criterion.

## 1. Open the live demo
Load the published Field Reasoning Console (link in the submission doc / README).
Click **"Load the challenge's example scenario"** — this sends exactly the
challenge brief's own example: *soil organic carbon 0.3%, low rainfall,
monoculture wheat, semi-arid.*

**What to point out:**
- The **O (Observed)** panel shows the four variables extracted from that one
  sentence — demonstrates variable extraction, not keyword-stuffing.
- The **A (Retrieved)** panel shows 2–3 real cited sources (with working
  links) that were actually similarity-ranked against the query, not
  hand-picked for the demo — click one to verify the citation is real.
- The **B (Reasoned)** panel shows a 4-variable causal chain
  (`crop_type → soil_organic_carbon → water_availability → species_richness`)
  — this is the "multi-metric reasoning, not single-variable answers"
  requirement made visible.
- The **C (Recommendations)** panel shows the structured output format from
  the brief exactly: what to do / why / mechanism / impacted metrics /
  expected impact / time horizon / evidence / confidence — and note the
  confidence levels differ per recommendation (High for the meta-analysis-
  backed cover crop finding, Medium for the single-region agroforestry study).

## 2. Show missing-data handling
Reset the conversation, then type just: *"Biodiversity is declining on my land."*
The system asks the exact three clarifying questions the brief's own example
specifies (soil organic carbon, rainfall pattern, land use type) — and no
recommendation panel populates yet, because the system won't guess.

## 3. Show multi-turn memory
Answer one variable at a time across three separate messages. Point out the
system never re-asks a question it already has an answer to, and the O panel
accumulates variables across turns.

## 4. Show contested evidence handling (the anti-fabrication story)
Click **"Ask about habitat fragmentation"**. The top retrieved source
(`kb006`) is explicitly a *contested* finding in real ecology (Fahrig 2017 vs.
Fletcher et al. 2018) — the system states the disagreement rather than
picking a side or inventing a consensus number, and the recommended action is
deliberately the lower-risk one ("protect total habitat area first"). This is
the single clearest piece of evidence that the system isn't fabricating
confidence it doesn't have.

## 5. Show the backend (if running locally / deployed)
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/knowledge/kb001 | python -m json.tool
```
Point out `/knowledge/{id}` returns the exact same record the demo's evidence
card displayed — the citation layer isn't decorative, it's the actual data
structure driving the answer.

## 6. Show the tests
```bash
cd backend && PYTHONPATH=. pytest tests/ -v
```
15 passing tests, including
`test_qualitative_kb_entries_never_produce_a_fabricated_percent_in_expected_impact`
— the anti-fabrication rule is enforced by CI, not just described in a doc.
