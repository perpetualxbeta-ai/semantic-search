"""
Lesson 1 — What is a vector, and how do we measure "closeness"?
No database, no AI model. Just numpy.

Run:  python lessons/01_vectors_101.py
"""
import numpy as np

# Imagine we describe animals with 3 made-up features:
#                  [is_pet, size, lives_in_water]
animals = {
    "dog":    np.array([0.9, 0.5, 0.0]),
    "cat":    np.array([0.9, 0.3, 0.0]),
    "wolf":   np.array([0.1, 0.6, 0.0]),
    "goldfish": np.array([0.8, 0.05, 1.0]),
    "shark":  np.array([0.0, 0.9, 1.0]),
}
# Each animal is now a VECTOR: an arrow pointing somewhere in 3-D space.
# Real embeddings work the same way, just with 384 (or 1536...) dimensions
# whose meanings were learned by a model instead of chosen by us.


def euclidean(a, b):
    """Straight-line distance between the arrow tips. Smaller = closer."""
    return np.linalg.norm(a - b)


def cosine_similarity(a, b):
    """
    Angle between the arrows, ignoring their length.
      1.0  -> same direction (very similar)
      0.0  -> perpendicular (unrelated)
     -1.0  -> opposite
    cosine DISTANCE (what pgvector's <=> returns) = 1 - cosine similarity.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


query = animals["dog"]
print("Which animals are most like a dog?\n")
print(f"{'animal':<10}{'cosine sim':>12}{'euclidean':>12}")
for name, vec in sorted(animals.items(), key=lambda kv: -cosine_similarity(query, kv[1])):
    print(f"{name:<10}{cosine_similarity(query, vec):>12.3f}{euclidean(query, vec):>12.3f}")

print("""
Takeaways:
 • 'cat' wins: it points in almost the same direction as 'dog'.
 • 'shark' loses: different on every feature.
 • Semantic search = turn the QUESTION into a vector, then find the
   stored vectors with the highest cosine similarity. That's it!

Try it: add a 'hamster' vector and guess its rank before running.
""")
