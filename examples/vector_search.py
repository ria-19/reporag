"""
THE SEARCH PROBLEM

Naive approach:
- Have 10,000 documents (10K embeddings)
- User query comes in
- Compute similarity with ALL 10K embeddings
- Return top 5

Complexity: O(n * d) where n=documents, d=dimensions
For 10K docs, 384 dims: 3.84 million operations PER QUERY

Can we do better? YES!
"""

import numpy as np
import time

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def naive_search(query_embedding, document_embeddings, k=5):
    """
    Brute force: Compare query to every document
    """
    similarities = []

    for i, doc_emb in enumerate(document_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb) # this is intensive dot product -> O(d)
        similarities.append((i, sim))
    
    # Sort and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

# Generate fake data
n_docs = 10_000
dim = 384
fake_docs = np.random.randn(n_docs, dim)
fake_query = np.random.randn(dim)

# Time it
start = time.time()
results = naive_search(fake_query, fake_docs, k=5)
elapsed = time.time() - start

print(f"Naive search: {elapsed*1000:.1f} ms for {n_docs} docs")
print(f"For 100K docs: ~{elapsed*10*1000:.0f} ms (too slow!)")


# Some optimzation
# Offline: preprocess v/|v| and put in a normalized db and use it -> upfront one time cost
# Online: preprocess each query before: q/|q| and then in loop, only dot product. 
# It's still O (n * d) but faster than naive by almost 2-3x

# Need better approach...

"""
FAISS: Facebook AI Similarity Search

Key Ideas:
1. Pre-process: Build an index (one-time cost)
2. Query-time: Use index for fast lookup

How?
- Cluster similar vectors together
- Search only relevant clusters (not all vectors)
- Trade-off: Slightly less accurate but MUCH faster

Types of Indices:
- Flat: Exact search (O(n)) - what we'll use
- IVF: Inverted File - clusters (O(√n))
- HNSW: Hierarchical graph (O(log n))
"""

"""
The fundamental workflow of FAISS:
 - Create an index, which is the data structure FAISS uses for fast searching.
 - Populate (or "add" to) the index with your dataset of vectors.
 - Perform a fast search on that index to find the nearest neighbors to a query vector.
"""

import faiss

# Create FAISS index
index = faiss.IndexFlatL2(dim) # L2 = Euclidean distance

# Add documents to index (one-time)
index.add(fake_docs.astype('float32'))  # FAISS needs float32; this is the "pre-processing" step or the "one-time cost." 

# Search (fast!)
start = time.time()
distances, indices = index.search(fake_query.reshape(1, -1).astype('float32'), k=5)
elapsed = time.time() - start

print(f"FAISS search: {elapsed*1000:.1f} ms for {n_docs} docs")
print(f"For 100K docs: ~{elapsed*10*1000:.0f} ms")

# For larger datasets, use IVF or HNSW for real speedup
# KEY INSIGHT:
# - Flat index: Good for <100K vectors (exact search)
# - IVF: Good for 100K-10M vectors (approximate)
# - HNSW: Good for 10M+ vectors (approximate, very fast)

"""
DISTANCE METRICS: Different ways to measure similarity

1. Euclidean Distance (L2)
   - Straight line distance
   - Sensitive to magnitude
   
2. Cosine Similarity
   - Angle between vectors
   - Ignores magnitude, only direction
   
3. Dot Product
   - Simple multiplication
   - Faster but less normalized

Which for RAG? Usually COSINE (ignore length, focus on direction)
"""

# Compare metrics
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

vec1 = np.array([1, 0])
vec2 = np.array([0.5, 0.5])
vec3 = np.array([2, 0])  # Same direction as vec1, different magnitude

print("Comparing distance metrics:")
print(f"vec1 = {vec1}, vec2 = {vec2}, vec3 = {vec3}")
print()

# Euclidean distance
print("Euclidean Distance (lower = more similar):")
print(f"  vec1 ↔ vec2: {euclidean_distances([vec1], [vec2])[0][0]:.3f}")
print(f"  vec1 ↔ vec3: {euclidean_distances([vec1], [vec3])[0][0]:.3f}")

# Cosine similarity
print("\nCosine Similarity (higher = more similar):")
print(f"  vec1 ↔ vec2: {cosine_similarity([vec1], [vec2])[0][0]:.3f}")
print(f"  vec1 ↔ vec3: {cosine_similarity([vec1], [vec3])[0][0]:.3f}")

# OBSERVE:
# - vec1 and vec3 are SAME DIRECTION (cosine = 1.0)
# - But different MAGNITUDE (euclidean = 1.0)
# 
# For text: Direction matters more than magnitude
# → Use COSINE for RAG

# PRACTICAL DECISION:
# FAISS IndexFlatL2 uses L2 distance (Euclidean)
# But we can normalize vectors → L2 distance ≈ Cosine similarity

embeddings = np.array([vec1, vec2, vec3])

# Normalize embeddings (unit vectors)
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("\nNormalized embeddings (for cosine-style search with L2):")
print(embeddings_normalized)

# Check similarity between normalized embeddings
print("\nCosine similarity between normalized docs:")
print(np.round(cosine_similarity(embeddings_normalized), 3))

print("\nEuclidean distances between normalized docs (approx inverse):")
print(np.round(euclidean_distances(embeddings_normalized), 3))