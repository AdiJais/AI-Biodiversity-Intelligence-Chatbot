"""
Standalone script to precompute and persist a FAISS index to disk, so a
production deployment doesn't re-embed the (small, static) knowledge base on
every cold start. Not required for local dev or tests - app/rag/retriever.py
builds the in-memory index lazily on first use either way. This script is
useful once the knowledge base grows large enough that startup latency
matters, or when you want a versioned, reviewable index artifact.

Usage:
    python -m app.rag.build_index --out backend/index/

Requires sentence-transformers + faiss-cpu to be installed (see
requirements.txt "optional, heavier dependencies").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.knowledge.loader import load_documents, documents_as_corpus_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="index", help="Output directory for the persisted index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers and faiss-cpu are required for this script. "
            "Install them with: pip install sentence-transformers faiss-cpu\n"
            f"Original error: {e}"
        )

    corpus = documents_as_corpus_text()
    docs = load_documents()

    print(f"Embedding {len(corpus)} knowledge base documents with {args.model} ...")
    model = SentenceTransformer(args.model)
    vectors = model.encode(corpus, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    # Normalize for cosine similarity via inner product index
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(out_dir / "kb.faiss"))
    with open(out_dir / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump([{"id": d["id"], "title": d["title"]} for d in docs], f, indent=2)

    print(f"Wrote {out_dir/'kb.faiss'} and {out_dir/'kb_metadata.json'}.")
    print("Note: app/rag/retriever.py currently builds its index in-memory at "
          "startup for simplicity; wire it to load this persisted index instead "
          "once the knowledge base is large enough that startup latency matters "
          "(see docs/ARCHITECTURE.md 'Scaling the retrieval layer').")


if __name__ == "__main__":
    main()
