"""
Lesson 4 — Why vector indexes matter (exact vs approximate search).

Inserts N random vectors, then times the same nearest-neighbour query:
  1. with no index   -> Postgres compares the query to EVERY row (exact)
  2. with HNSW index -> walks a graph of neighbours (approximate, fast)
and checks how many of the true top-10 the index found ("recall").

Run:  python lessons/04_indexes.py            (default 20,000 rows)
      python lessons/04_indexes.py 100000
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_search.db import connect  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
DIM = 384
rng = np.random.default_rng(42)

conn = connect()
conn.execute("DROP TABLE IF EXISTS bench")
conn.execute(f"CREATE TABLE bench (id int PRIMARY KEY, embedding vector({DIM}))")

print(f"Inserting {N:,} random {DIM}-d vectors ...")
data = rng.normal(size=(N, DIM)).astype(np.float32)
with conn.cursor().copy("COPY bench (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
    copy.set_types(["int4", "vector"])
    for i, v in enumerate(data):
        copy.write_row((i, v))

query = rng.normal(size=DIM).astype(np.float32)
SQL = "SELECT id FROM bench ORDER BY embedding <=> %s LIMIT 10"


def timed(label):
    t = time.perf_counter()
    for _ in range(20):
        ids = [r[0] for r in conn.execute(SQL, (query,)).fetchall()]
    ms = (time.perf_counter() - t) / 20 * 1000
    plan = conn.execute("EXPLAIN " + SQL, (query,)).fetchall()
    scan = next(r[0] for r in plan if "Scan" in r[0]).strip()
    print(f"{label:<12} {ms:7.2f} ms/query   plan: {scan}")
    return ids


exact = timed("No index")

print("Building HNSW index ...")
t = time.perf_counter()
conn.execute("CREATE INDEX ON bench USING hnsw (embedding vector_cosine_ops)")
print(f"  built in {time.perf_counter() - t:.1f}s")
conn.execute("ANALYZE bench")

approx = timed("HNSW")
print(f"\nRecall@10: {len(set(exact) & set(approx))}/10 of the exact answers found")

conn.execute("SET hnsw.ef_search = 200")  # search more of the graph: slower, more accurate
approx2 = timed("HNSW ef=200")
print(f"Recall@10: {len(set(exact) & set(approx2))}/10")

conn.execute("DROP TABLE bench")
print("""
Takeaways:
 • Indexes trade a little accuracy for a lot of speed. Tune with hnsw.ef_search.
 • Under ~10k rows, the planner may skip the index — a full scan is already fast.
 • Random vectors are the HARDEST case for HNSW; real embeddings cluster, so recall is usually better.
""")
