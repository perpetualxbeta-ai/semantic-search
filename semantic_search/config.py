"""Settings, read from environment variables (or a .env file)."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Tiny .env loader so we don't need an extra dependency."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/semantic_search"
)

# "model" = a real sentence-transformer (semantic meaning)
# "toy"   = hashed bag-of-words (offline, no download, keyword-ish) — for experiments
EMBEDDER = os.getenv("EMBEDDER", "model")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# all-MiniLM-L6-v2 turns any text into a list of 384 numbers.
# The column in Postgres must have the same size: vector(384).
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
