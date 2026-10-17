-- Lesson 3 — pgvector in plain SQL, with tiny 3-D vectors you can read.
-- Run:  psql postgresql://postgres:postgres@localhost:5432/semantic_search -f lessons/03_pgvector_sql.sql

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS toy_animals;
CREATE TABLE toy_animals (
    name      text,
    embedding vector(3)          -- [is_pet, size, lives_in_water]
);

INSERT INTO toy_animals VALUES
  ('dog',      '[0.9, 0.5, 0.0]'),
  ('cat',      '[0.9, 0.3, 0.0]'),
  ('wolf',     '[0.1, 0.6, 0.0]'),
  ('goldfish', '[0.8, 0.05, 1.0]'),
  ('shark',    '[0.0, 0.9, 1.0]');

-- The three distance operators. For all of them, SMALLER = MORE SIMILAR.
SELECT name,
       embedding <-> '[0.9,0.5,0]'      AS l2_distance,       -- Euclidean
       embedding <=> '[0.9,0.5,0]'      AS cosine_distance,   -- 1 - cosine similarity
       (embedding <#> '[0.9,0.5,0]') * -1 AS inner_product
FROM toy_animals
ORDER BY embedding <=> '[0.9,0.5,0]';

-- "Nearest neighbour" query: the core of semantic search.
SELECT name FROM toy_animals ORDER BY embedding <=> '[0.0, 0.8, 0.9]' LIMIT 2;  -- something big that swims

-- Handy helpers
SELECT vector_dims(embedding), vector_norm(embedding) FROM toy_animals LIMIT 1;
SELECT avg(embedding) AS average_pet FROM toy_animals WHERE name IN ('dog','cat');

DROP TABLE toy_animals;
