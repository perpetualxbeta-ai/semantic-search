"""
Command line interface.

    python -m semantic_search.cli init              # create table + index
    python -m semantic_search.cli ingest            # embed & load data/sample_docs.jsonl
    python -m semantic_search.cli search "how do I fix a flat bike tire"
    python -m semantic_search.cli compare "automobile repair"   # semantic vs keyword
    python -m semantic_search.cli similar 3         # docs like document #3
    python -m semantic_search.cli chat              # interactive search loop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, db
from .embeddings import embed, embed_one, get_embedder

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_docs.jsonl"


def _print_hits(hits: list[db.Hit], label: str = "similarity") -> None:
    if not hits:
        print("  (no results)")
    for rank, h in enumerate(hits, 1):
        bar = "█" * max(0, int(h.similarity * 20))
        print(f"  {rank}. [{h.id:>3}] {h.title:<38} {label}={h.similarity:.3f} {bar}")


def cmd_init(args) -> None:
    dim = get_embedder().dim
    with db.connect() as conn:
        db.init_schema(conn, dim=dim, reset=args.reset)
    print(f"✓ table 'documents' ready (vector({dim}), HNSW cosine index)")


def cmd_ingest(args) -> None:
    path = Path(args.file)
    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"Embedding {len(docs)} documents with EMBEDDER={config.EMBEDDER} ...")
    vectors = embed([f"{d['title']}. {d['content']}" for d in docs])
    print(f"  each document -> vector of {vectors.shape[1]} numbers, e.g. {vectors[0][:5].round(3)} ...")
    with db.connect() as conn:
        db.init_schema(conn, dim=vectors.shape[1], reset=args.reset)
        n = db.insert_documents(conn, docs, vectors)
        print(f"✓ inserted {n} rows (table now has {db.count(conn)})")


def cmd_search(args) -> None:
    with db.connect() as conn:
        print(f'\nSemantic search: "{args.query}"')
        _print_hits(db.search(conn, embed_one(args.query), k=args.k))


def cmd_compare(args) -> None:
    with db.connect() as conn:
        print(f'\nQuery: "{args.query}"\n\n🧠 Semantic (pgvector, meaning):')
        _print_hits(db.search(conn, embed_one(args.query), k=args.k))
        print("\n🔤 Keyword (Postgres full-text, exact words):")
        _print_hits(db.keyword_search(conn, args.query, k=args.k), label="rank")


def cmd_similar(args) -> None:
    with db.connect() as conn:
        print(f"\nDocuments most similar to #{args.id}:")
        _print_hits(db.similar_to(conn, args.id, k=args.k))


def cmd_chat(args) -> None:
    print("Type a question (empty line to quit).")
    with db.connect() as conn:
        while True:
            try:
                q = input("\n🔎 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            _print_hits(db.search(conn, embed_one(q), k=args.k))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semantic_search")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create the table and index")
    s.add_argument("--reset", action="store_true", help="drop the table first")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("ingest", help="embed a JSONL file and store it")
    s.add_argument("file", nargs="?", default=str(DEFAULT_DATA))
    s.add_argument("--reset", action="store_true", help="drop the table first")
    s.set_defaults(func=cmd_ingest)

    for name, fn, help_ in [("search", cmd_search, "semantic search"),
                            ("compare", cmd_compare, "semantic vs keyword search")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=5)
        s.set_defaults(func=fn)

    s = sub.add_parser("similar", help="'more like this' for a document id")
    s.add_argument("id", type=int)
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_similar)

    s = sub.add_parser("chat", help="interactive search loop")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_chat)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:  # friendly errors for beginners
        if "connection" in str(e).lower():
            sys.exit(f"Can't reach Postgres at {config.DATABASE_URL}\n"
                     "Is it running? Try:  docker compose up -d")
        raise


if __name__ == "__main__":
    main()
