# Semantic Search with Python + PostgreSQL (pgvector)

A small, heavily-commented project for **learning how vectors and semantic search work**.
Ask a question in plain English and it finds documents with the same *meaning* —
even when they share no words.

```
Query: "my automobile won't turn on"          (illustrative — your scores will differ)

🧠 Semantic (pgvector)                      🔤 Keyword (full-text)
 1. Car engine won't start                    (no results — no shared words)
 2. Changing your vehicle's motor oil
```

## How it works (the 30-second version)

```
 text ──► embedding model ──► [0.12, -0.03, 0.88, … 384 numbers] ──► stored in Postgres (vector column)

 question ──► same model ──► query vector ──► "ORDER BY embedding <=> query LIMIT 5" ──► closest meanings
```

1. **Embed** — a small neural network (`all-MiniLM-L6-v2`) turns each text into a vector of 384 numbers.
   Similar meanings → vectors pointing in similar directions.
2. **Store** — pgvector adds a `vector(384)` column type to Postgres.
3. **Search** — embed the question, then ask Postgres for the nearest vectors using cosine distance (`<=>`).
4. **Index** — an HNSW index makes that fast at millions of rows.

## Setup

You need Python 3.10+ and Docker.

```bash
git clone https://github.com/perpetualxbeta-ai/semantic-search
cd semantic-search

docker compose up -d                 # Postgres 16 + pgvector on localhost:5432

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Try it

```bash
python -m semantic_search.cli ingest --reset        # embed + load 32 sample docs
python -m semantic_search.cli search "how do I fix a flat bike tyre"
python -m semantic_search.cli compare "automobile repair"   # semantic vs keyword side by side
python -m semantic_search.cli similar 4             # "more like this" for doc #4
python -m semantic_search.cli chat                  # interactive
```

The first run downloads the model (~90 MB). Want zero downloads? Set `EMBEDDER=toy`
(in `.env` or the shell) — a word-hashing embedder that behaves like keyword search.
Comparing the two is a great way to *feel* what a real model adds.

### Fun queries to try
- `"pet that purrs"` → finds the cat article without the word "cat"
- `"good place for cheap local food"` → hawker centres
- `"investing without picking stocks"` → index funds
- `"I feel anxious"` → reducing stress
- `"database"` vs `"make my queries faster"` — compare with `compare`

## Learning path

| # | File | What you learn |
|---|------|----------------|
| 1 | `lessons/01_vectors_101.py` | Vectors as arrows; cosine similarity vs Euclidean distance — numpy only |
| 2 | `lessons/02_real_embeddings.py` | What a real model produces; similarity matrix of sentences |
| 3 | `lessons/03_pgvector_sql.sql` | pgvector in raw SQL with readable 3-D vectors: `<->`, `<=>`, `<#>` |
| 4 | `lessons/04_indexes.py` | Exact vs approximate search, HNSW speed-up, recall, `ef_search` tuning |
| 5 | `semantic_search/` | The real app: `embeddings.py` → `db.py` → `cli.py` |

Run lesson 3 with:
```bash
psql postgresql://postgres:postgres@localhost:5432/semantic_search -f lessons/03_pgvector_sql.sql
```

## pgvector cheat sheet

| Operator | Meaning | Index ops class |
|---|---|---|
| `a <-> b` | Euclidean (L2) distance | `vector_l2_ops` |
| `a <=> b` | Cosine distance = 1 − cosine similarity | `vector_cosine_ops` |
| `a <#> b` | **Negative** inner product | `vector_ip_ops` |

All are "smaller = closer", so you always `ORDER BY … ASC`.

## Project layout

```
semantic_search/
  config.py       settings from env / .env
  embeddings.py   text → vectors (real model or toy)
  db.py           schema, insert, semantic/keyword/"similar" search
  cli.py          command line
data/sample_docs.jsonl   32 short docs (lots of synonyms on purpose)
lessons/                 step-by-step learning scripts
tests/                   pytest, runs offline with EMBEDDER=toy
```

## Use your own data

Make a JSONL file, one doc per line: `{"title": "...", "content": "..."}`, then

```bash
python -m semantic_search.cli ingest my_docs.jsonl --reset
```

## Next experiments

- **Chunking** — split long articles into ~200-word pieces before embedding.
- **Hybrid search** — combine `db.search` and `db.keyword_search` scores (Reciprocal Rank Fusion).
- **Metadata filters** — add a `category` column and `WHERE category = 'cooking'` before `ORDER BY`.
- **Other models** — try `BAAI/bge-small-en-v1.5` (set `MODEL_NAME`; re-ingest with `--reset`).
- **RAG** — feed the top hits to an LLM to answer questions from your documents.

## Tests

```bash
EMBEDDER=toy pytest -q      # needs the docker DB running
```
