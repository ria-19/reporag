"""
SimpleRAG: Advanced Exercises
==============================

Build on the core SimpleRAG implementation with production-grade features.
Each exercise teaches a specific concept used in real RAG systems.

Prerequisites: Complete simple_rag.py first
"""

import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==============================================================================
# EXERCISE 1: Metadata Filtering
# ==============================================================================
"""
PROBLEM: Sometimes you want to search only specific documents
EXAMPLE: "Find articles about Python, but only from 2024"

SOLUTION: Pre-filter documents by metadata before searching
"""

def search_with_filter(self, query: str, category: str = None, k: int = 3):
    """
    Search only within documents matching a category.
    
    How it works:
    1. Filter documents by metadata FIRST
    2. Search only within filtered set
    3. Return top-k from filtered results
    
    Why this matters:
    - Users often need domain-specific results (e.g., "legal" vs "medical")
    - Reduces search space for faster queries
    - Improves precision by removing irrelevant categories
    """
    if len(self.documents) == 0:
        return []
    
    # Pre-filter by category if specified
    if category:
        filtered_indices = [
            i for i, doc in enumerate(self.documents)
            if doc.metadata.get("category") == category
        ]
        if not filtered_indices:
            return []  # No documents match the filter
        
        filtered_docs = [self.documents[i] for i in filtered_indices]
        filtered_embeddings = self.embeddings[filtered_indices]
    else:
        filtered_docs = self.documents
        filtered_embeddings = self.embeddings
    
    # Now search within filtered set
    query_embedding = self.embedding_model.encode(query)
    similarities = []
    
    for doc_embedding in filtered_embeddings:
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append(similarity)
    
    # Get top-k from filtered results
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    original_indices = filtered_indices if category else list(range(len(self.documents)))
    
    return [(filtered_docs[i], similarities[i]) for i in top_k_indices]


"""
PRODUCTION NOTE:
In real systems, metadata filtering happens at the vector database level.
Databases like Pinecone, Weaviate, and Qdrant support filtered queries:
    
    results = index.query(
        vector=query_embedding,
        filter={"category": {"$eq": "medical"}},
        top_k=10
    )

This pushes filtering down to the database, making it much faster.
"""


# ==============================================================================
# EXERCISE 2: Re-ranking with Cross-Encoders
# ==============================================================================
"""
PROBLEM: Bi-encoders (like all-MiniLM-L6-v2) embed query and docs separately.
         They're fast but miss subtle query-document interactions.

SOLUTION: Two-stage retrieval
    Stage 1: Bi-encoder retrieves top-N candidates (fast, broad recall)
    Stage 2: Cross-encoder re-ranks top-N (slow, high precision)
"""

class ReRanker:
    """
    Re-ranks search results using a cross-encoder model.
    
    Bi-Encoder vs Cross-Encoder:
    
    Bi-Encoder:
    ✓ Embeds query and docs separately → fast cosine similarity
    ✓ Can pre-compute document embeddings
    ✗ Misses query-document interaction
    
    Cross-Encoder:
    ✓ Processes query + document together → highly accurate
    ✓ Captures nuanced relevance
    ✗ Cannot pre-compute (must run for each query)
    ✗ Much slower (requires N forward passes for N documents)
    
    Best practice: Retrieve 20-50 with bi-encoder, re-rank top-10 with cross-encoder
    """
    
    def __init__(self):
        # Cross-encoder trained on MS MARCO (passage ranking dataset)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query: str, results: List[Tuple], top_k: int = None):
        """
        Re-rank retrieved results using cross-encoder scores.
        
        Args:
            query: The search query
            results: List of (Document, bi_encoder_score) tuples
            top_k: Return only top-k after re-ranking (optional)
        
        Returns:
            Re-ranked list of (Document, cross_encoder_score) tuples
        """
        if not results:
            return []
        
        # Prepare pairs for cross-encoder: [(query, doc1), (query, doc2), ...]
        pairs = [(query, doc.content) for doc, _ in results]
        
        # Get cross-encoder scores (measures true query-document relevance)
        cross_scores = self.reranker.predict(pairs)
        
        # Re-rank by cross-encoder scores
        reranked = sorted(
            zip([doc for doc, _ in results], cross_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return reranked[:top_k] if top_k else reranked


"""
USAGE EXAMPLE:

# Stage 1: Fast retrieval with bi-encoder
initial_results = rag.search(query, k=20)

# Stage 2: Precise re-ranking with cross-encoder
reranker = ReRanker()
final_results = reranker.rerank(query, initial_results, top_k=5)

This gives you the best of both worlds: speed + accuracy!
"""


# ==============================================================================
# EXERCISE 3: Alternative Similarity Metrics
# ==============================================================================
"""
PROBLEM: Not all similarity metrics work the same way
QUESTION: When should you use cosine vs dot product vs euclidean?

ANSWER: Depends on how your embedding model was trained!
"""

def search_with_metric(self, query: str, k: int = 3, metric: str = "cosine"):
    """
    Compare different similarity metrics for vector search.
    
    COSINE SIMILARITY:
    - Measures angle between vectors (ignores magnitude)
    - Range: -1 to 1 (usually 0 to 1 for embeddings)
    - Best for: Most embedding models (normalized vectors)
    - Formula: (A·B) / (|A| × |B|)
    
    DOT PRODUCT:
    - Measures alignment AND magnitude
    - Range: -∞ to +∞
    - Best for: Models trained with dot product objective
    - Formula: A·B
    
    EUCLIDEAN DISTANCE:
    - Measures straight-line distance in vector space
    - Range: 0 to +∞ (smaller = more similar)
    - Best for: When you care about absolute position, not just direction
    - Formula: |A - B|
    
    Rule of thumb:
    - Use cosine for text embeddings (most common)
    - Use dot product if embeddings are already normalized
    - Use euclidean for specialized cases (rare)
    """
    if len(self.documents) == 0:
        return []
    
    query_embedding = self.embedding_model.encode(query)
    scores = []
    
    for doc_embedding in self.embeddings:
        if metric == "cosine":
            score = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
        elif metric == "dot":
            score = np.dot(query_embedding, doc_embedding)
        elif metric == "euclidean":
            # Negative distance so higher = better (consistent with other metrics)
            score = -np.linalg.norm(query_embedding - doc_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine', 'dot', or 'euclidean'")
        
        scores.append(score)
    
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return [(self.documents[i], scores[i]) for i in top_k_indices]


"""
PRO TIP: Check your embedding model's documentation!
Some models (like those from OpenAI) are trained with dot product,
so using cosine similarity would be suboptimal.
"""


# ==============================================================================
# EXERCISE 4: Caching Strategies
# ==============================================================================
"""
PROBLEM: Repeated queries waste compute and time
SOLUTION: Cache at different levels of the pipeline

Three levels of caching:
1. Embedding Cache: Save query → vector (avoid re-encoding)
2. Retrieval Cache: Save query → document IDs (skip search)
3. Answer Cache: Save query → final answer (skip LLM call)

Tradeoffs:
- Level 1: Always safe, small memory, modest speedup
- Level 2: Requires invalidation when docs change
- Level 3: Largest speedup, but answers may become stale
"""

class RAGCache:
    """
    Multi-level cache for RAG operations.
    
    Invalidation is the hard part:
    - When should you clear the cache?
    - When documents change? When model updates? After N hours?
    
    Production strategy:
    - Set TTL (time-to-live) for each cache level
    - Use cache keys that include document version/hash
    - Provide manual cache clearing for admins
    """
    
    def __init__(self):
        self.embedding_cache = {}      # query → vector
        self.retrieval_cache = {}      # query → doc IDs
        self.answer_cache = {}         # query → final answer
    
    def get_cached_answer(self, query: str) -> Dict:
        """Check if we already answered this exact query."""
        return self.answer_cache.get(query)
    
    def cache_answer(self, query: str, result: Dict):
        """Store a complete query result."""
        self.answer_cache[query] = result
    
    def clear_all(self):
        """Invalidate all caches (call when documents change)."""
        self.embedding_cache.clear()
        self.retrieval_cache.clear()
        self.answer_cache.clear()
        print("🗑️ All caches cleared")


"""
USAGE WITH RAG:

cache = RAGCache()

def query_with_cache(self, question: str, k: int = 3):
    # Check cache first
    cached = cache.get_cached_answer(question)
    if cached:
        print("⚡ Using cached result")
        return cached
    
    # Run normal query
    result = self.query(question, k=k)
    
    # Store in cache
    cache.cache_answer(question, result)
    
    return result
"""


# ==============================================================================
# EXERCISE 5: Batch Processing
# ==============================================================================
"""
PROBLEM: Processing queries one-at-a-time is inefficient
SOLUTION: Batch multiple operations together

Why batching matters:
- GPUs are designed for parallel operations
- Embedding models process batches faster than N individual calls
- Vector databases can search multiple queries at once
- LLMs can batch requests for higher throughput

Example speedup:
- Sequential: 100 queries × 50ms = 5000ms
- Batched: 100 queries / 10 per batch × 100ms = 1000ms (5x faster!)

Tradeoff: Slightly higher latency per query, massively higher throughput
"""

def batch_query(self, queries: List[str], k: int = 3) -> List[Dict]:
    """
    Process multiple queries efficiently in a batch.
    
    Key optimization: Embed all queries at once
    Instead of: [embed(q1), embed(q2), ..., embed(qN)]
    Do this: embed([q1, q2, ..., qN])  # One call!
    
    This is 3-10x faster depending on batch size.
    """
    if not queries:
        return []
    
    print(f"📦 Processing batch of {len(queries)} queries...")
    
    # Batch embed all queries at once (FAST!)
    query_embeddings = self.embedding_model.encode(queries)
    
    results = []
    for i, query in enumerate(queries):
        query_emb = query_embeddings[i]
        
        # Calculate similarities for this query
        similarities = []
        for doc_emb in self.embeddings:
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            similarities.append(sim)
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        query_results = [(self.documents[idx], similarities[idx]) for idx in top_k_indices]
        
        # Build context and generate answer
        context_parts = [f"[{doc.metadata.get('title')}]\n{doc.content}" 
                        for doc, _ in query_results]
        context = "\n\n".join(context_parts)
        answer = self.generate_answer(query, context)
        
        results.append({
            "question": query,
            "answer": answer,
            "sources": [doc.metadata.get('title') for doc, _ in query_results],
            "context": context_parts
        })
    
    print(f"✓ Completed {len(queries)} queries")
    return results


"""
PRODUCTION PATTERN: Request Batching

Instead of processing each user query immediately, collect them briefly:

batch_queue = []
async def handle_query(query):
    batch_queue.append(query)
    
    # Wait for more queries OR timeout
    await wait_for_batch(max_size=32, timeout_ms=50)
    
    # Process entire batch together
    results = rag.batch_query(batch_queue)
    
    # Return result to each user
    return results[batch_queue.index(query)]

This adds ~50ms latency but increases throughput by 5-10x!
Perfect for high-traffic applications.
"""


# ==============================================================================
# COMPLETE EXAMPLE: Putting it all together
# ==============================================================================

def advanced_rag_demo():
    """
    Demonstrates all exercises working together.
    """
    from simple_rag import SimpleRAG  # Import base RAG
    
    rag = SimpleRAG()
    
    # Add documents with rich metadata
    rag.add_document(
        "Python is great for ML with libraries like PyTorch.",
        {"title": "Python", "category": "programming", "date": "2024"}
    )
    rag.add_document(
        "JavaScript is the language of the web and browsers.",
        {"title": "JavaScript", "category": "programming", "date": "2024"}
    )
    rag.add_document(
        "Machine learning models learn patterns from data.",
        {"title": "ML Overview", "category": "ai", "date": "2024"}
    )
    
    # Exercise 1: Filtered search
    print("\n🔍 Filtered Search (programming only):")
    results = search_with_filter(rag, "best language", category="programming", k=2)
    for doc, score in results:
        print(f"   {doc.metadata['title']}: {score:.3f}")
    
    # Exercise 2: Re-ranking
    print("\n🎯 Re-ranking with cross-encoder:")
    reranker = ReRanker()
    initial = rag.search("programming languages", k=3)
    reranked = reranker.rerank("programming languages", initial, top_k=2)
    for doc, score in reranked:
        print(f"   {doc.metadata['title']}: {score:.3f}")
    
    # Exercise 3: Different metrics
    print("\n📊 Comparing similarity metrics:")
    for metric in ["cosine", "dot", "euclidean"]:
        results = search_with_metric(rag, "Python", k=1, metric=metric)
        doc, score = results[0]
        print(f"   {metric}: {doc.metadata['title']} ({score:.3f})")
    
    # Exercise 4: Caching
    print("\n⚡ Caching demonstration:")
    cache = RAGCache()
    # First call - slow
    result1 = rag.query("What is Python?")
    cache.cache_answer("What is Python?", result1)
    # Second call - fast
    result2 = cache.get_cached_answer("What is Python?")
    print(f"   Cache hit: {result2 is not None}")
    
    # Exercise 5: Batch processing
    print("\n📦 Batch processing:")
    queries = ["What is Python?", "Tell me about ML", "JavaScript uses?"]
    results = batch_query(rag, queries, k=2)
    print(f"   Processed {len(results)} queries in one batch")


if __name__ == "__main__":
    print("=" * 70)
    print("SimpleRAG: Advanced Exercises")
    print("=" * 70)
    advanced_rag_demo()