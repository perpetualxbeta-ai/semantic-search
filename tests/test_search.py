"""
End-to-end test against a real Postgres, using the offline toy embedder
(so it runs without downloading a model).

    EMBEDDER=toy pytest -q
"""
import os

os.environ.setdefault("EMBEDDER", "toy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from semantic_search import db  # noqa: E402
from semantic_search.embeddings import ToyEmbedder  # noqa: E402

DOCS = [
    {"title": "Bike repair", "content": "patch a flat bicycle tube"},
    {"title": "Pancakes", "content": "whisk flour milk and egg for pancakes"},
    {"title": "Black holes", "content": "gravity so strong light cannot escape"},
]


@pytest.fixture
def conn():
    try:
        c = db.connect()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not available: {e}")
    c.execute("DROP TABLE IF EXISTS documents")
    db.init_schema(c, dim=384)
    yield c
    c.execute("DROP TABLE IF EXISTS documents")
    c.close()


def test_toy_vectors_are_unit_length():
    v = ToyEmbedder().embed(["hello world", "another sentence"])
    assert v.shape == (2, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_insert_and_search(conn):
    emb = ToyEmbedder()
    db.insert_documents(conn, DOCS, emb.embed([d["content"] for d in DOCS]))
    assert db.count(conn) == 3

    hits = db.search(conn, emb.embed(["flour and milk pancakes"])[0], k=3)
    assert hits[0].title == "Pancakes"
    assert hits[0].similarity > hits[1].similarity  # ranked best-first


def test_similar_excludes_itself(conn):
    emb = ToyEmbedder()
    db.insert_documents(conn, DOCS, emb.embed([d["content"] for d in DOCS]))
    hits = db.similar_to(conn, 1, k=5)
    assert all(h.id != 1 for h in hits)
    assert len(hits) == 2


def test_keyword_search(conn):
    emb = ToyEmbedder()
    db.insert_documents(conn, DOCS, emb.embed([d["content"] for d in DOCS]))
    hits = db.keyword_search(conn, "gravity")
    assert [h.title for h in hits] == ["Black holes"]
