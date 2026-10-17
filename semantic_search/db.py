"""
Everything that talks to PostgreSQL + pgvector.

pgvector adds a `vector(N)` column type and three distance operators:
    <->  Euclidean (L2) distance       smaller = closer
    <#>  negative inner product        smaller = closer
    <=>  cosine distance (1 - cosine)  smaller = closer   <- we use this one
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from . import config


@dataclass
class Hit:
    id: int
    title: str
    content: str
    similarity: float  # 1.0 = identical direction, 0 = unrelated


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets us pass numpy arrays straight into SQL
    return conn


def init_schema(conn: psycopg.Connection, dim: int = config.EMBEDDING_DIM, reset: bool = False) -> None:
    if reset:
        conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id         bigserial PRIMARY KEY,
            title      text NOT NULL,
            content    text NOT NULL,
            embedding  vector({dim}) NOT NULL
        )
        """
    )
    # HNSW = an approximate-nearest-neighbour index (a graph of "close" vectors).
    # Without it, Postgres compares the query to EVERY row (fine for thousands,
    # slow for millions). vector_cosine_ops makes it serve the <=> operator.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS documents_embedding_hnsw "
        "ON documents USING hnsw (embedding vector_cosine_ops)"
    )


def insert_documents(conn: psycopg.Connection, docs: list[dict], vectors: np.ndarray) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO documents (title, content, embedding) VALUES (%s, %s, %s)",
            [(d["title"], d["content"], v) for d, v in zip(docs, vectors)],
        )
    return len(docs)


def search(conn: psycopg.Connection, query_vec: np.ndarray, k: int = 5) -> list[Hit]:
    rows = conn.execute(
        """
        SELECT id, title, content, 1 - (embedding <=> %(q)s) AS similarity
        FROM documents
        ORDER BY embedding <=> %(q)s      -- nearest vectors first
        LIMIT %(k)s
        """,
        {"q": query_vec, "k": k},
    ).fetchall()
    return [Hit(*r) for r in rows]


def keyword_search(conn: psycopg.Connection, query: str, k: int = 5) -> list[Hit]:
    """Old-school full-text search, for comparison. Matches words, not meaning."""
    rows = conn.execute(
        """
        SELECT id, title, content,
               ts_rank(to_tsvector('english', title || ' ' || content),
                       plainto_tsquery('english', %(q)s)) AS score
        FROM documents
        WHERE to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %(q)s)
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        {"q": query, "k": k},
    ).fetchall()
    return [Hit(*r) for r in rows]


def similar_to(conn: psycopg.Connection, doc_id: int, k: int = 5) -> list[Hit]:
    """'More like this': use a stored document's own vector as the query."""
    rows = conn.execute(
        """
        SELECT d.id, d.title, d.content, 1 - (d.embedding <=> src.embedding)
        FROM documents d, (SELECT embedding FROM documents WHERE id = %(id)s) src
        WHERE d.id <> %(id)s
        ORDER BY d.embedding <=> src.embedding
        LIMIT %(k)s
        """,
        {"id": doc_id, "k": k},
    ).fetchall()
    return [Hit(*r) for r in rows]


def count(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM documents").fetchone()[0]
