"""
Lesson 2 — Real embeddings from a language model.
The model turns sentences into 384-number vectors. We never decide what the
numbers mean; we only check which sentences end up close together.

Run:  python lessons/02_real_embeddings.py
      (first run downloads ~90 MB model)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from semantic_search.embeddings import embed  # noqa: E402

sentences = [
    "My car won't start in the morning",
    "The automobile engine fails to turn over",   # same meaning, different words!
    "How do I bake a chocolate cake?",
    "Recipe for a rich cocoa dessert",            # same meaning, different words!
    "The stock market fell sharply today",
]

vectors = embed(sentences)
print(f"Shape: {vectors.shape}  -> {len(sentences)} sentences x {vectors.shape[1]} numbers each")
print(f"First 8 numbers of sentence 0: {np.round(vectors[0][:8], 3)}\n")

# Vectors are normalised to length 1, so a dot product IS cosine similarity.
sim = vectors @ vectors.T

print("Cosine similarity matrix (1.0 = same meaning):\n")
short = [s[:28] for s in sentences]
print(" " * 30 + "".join(f"{i:>7}" for i in range(len(sentences))))
for i, row in enumerate(sim):
    print(f"{i} {short[i]:<28}" + "".join(f"{v:>7.2f}" for v in row))

print("""
Look at (0,1) and (2,3): high scores even though they share almost no words.
That's what keyword search can't do — and why we store these vectors in pgvector.
""")
