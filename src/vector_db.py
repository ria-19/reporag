"""
Vector Database
Goal: Efficient storage and retrieval at scale
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
from chunking import CodeAwareChunker

class VectorDB:
    """
    Vector database
    
    Features:
    - FAISS for fast search
    - Persistent storage
    - Metadata filtering
    - Incremental updates
    """
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):  # Factory Pattern
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "flat":
            # Exact search (good for <100K vectors)
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            # Approximate search (good for 100K-10M vectors)
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Store metadata separately (FAISS only stores vectors)
        # This store links the numerical vector IDs (from FAISS) back to the actual text content and source metadata.
        self.metadata_store: List[Dict] = []  
        self.id_counter = 0
    
    # Pre-processing Pipeline
    def add_documents(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add documents to database
        
        Args:
            embeddings: (n, dimension) array
            metadata: List of metadata dicts
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)    #key pre-processing step
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add metadata with IDs
        for i, meta in enumerate(metadata):
            meta_with_id = meta.copy()
            meta_with_id['_id'] = self.id_counter
            self.metadata_store.append(meta_with_id)
            self.id_counter += 1
        
        print(f"✅ Added {len(embeddings)} vectors (total: {self.index.ntotal})")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, filters: Dict = None) -> List[Tuple[Dict, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Metadata filters (e.g., {'source': 'github'})
        
        Returns:
            List of (metadata, score) tuples
        """
        # Normalize query
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        if filters:
            # Need to over-fetch and filter
            k_fetch = min(k * 10, self.index.ntotal)
        else:
            k_fetch = k
        
        distances, indices = self.index.search(query_embedding, k_fetch)
        
        # Get results with metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            metadata = self.metadata_store[idx]
            
            # Apply filters
            if filters:
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            # Convert L2 distance to similarity score (0-1)
            similarity = 1 / (1 + dist)
            results.append((metadata, similarity))
            
            if len(results) == k:
                break
        
        return results
    
    # Serialization Pattern
    def save(self, path: str):
        """Save index and metadata to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss")) # FAISS Serialization
        
        # Save metadata
        with open(path / "metadata.pkl", 'wb') as f: # Python State Serialization
            pickle.dump({
                'metadata_store': self.metadata_store,
                'id_counter': self.id_counter,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        print(f"✅ Saved to {path}")
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load metadata
        with open(path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data['metadata_store']
            self.id_counter = data['id_counter']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        print(f"✅ Loaded from {path} ({self.index.ntotal} vectors)")


# ============================================================
# INDEXING PIPELINE
# ============================================================

class IndexingPipeline:
    """
    Complete pipeline: Documents → Embeddings → Vector DB
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.vector_db = VectorDB(dimension=self.dimension)
        self.chunker = CodeAwareChunker()
    
    # Offline Part
    def index_documents(self, documents: List[Dict], batch_size: int = 32):
        """
        Index documents into vector database
        
        Steps:
        1. Chunk documents
        2. Generate embeddings (batched)
        3. Add to vector database
        """
        print(f"\n{'='*60}")
        print("INDEXING PIPELINE")
        print(f"{'='*60}\n")
        
        # Step 1: Chunk
        print("1. Chunking documents...")
        chunks = self.chunker.chunk_all(documents)
        print(f"   Total chunks: {len(chunks)}")
        
        # Step 2: Generate embeddings (batched for efficiency)
        print("\n2. Generating embeddings...")
        all_embeddings = []
        all_metadata = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['content'] for chunk in batch]
            metadata = [chunk['metadata'] for chunk in batch]
            
            # Encode batch
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size
            )
            
            all_embeddings.append(embeddings)
            all_metadata.extend(metadata)
        
        # Combine batches
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"   Embeddings shape: {all_embeddings.shape}")
        
        # Step 3: Add to vector database
        print("\n3. Adding to vector database...")
        self.vector_db.add_documents(all_embeddings, all_metadata)
        
        print(f"\n{'='*60}")
        print("✅ INDEXING COMPLETE")
        print(f"{'='*60}\n")
    
    # Online Part
    def query(self, question: str, k: int = 5, filters: Dict = None):
        """Query the indexed knowledge base"""
        # Embed question
        query_embedding = self.model.encode(question)
        
        # Search
        results = self.vector_db.search(query_embedding, k=k, filters=filters)
        
        return results
    
    def save(self, path: str):
        """Save entire pipeline"""
        self.vector_db.save(path)
    
    def load(self, path: str):
        """Load entire pipeline"""
        self.vector_db.load(path)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Create test documents
    test_docs = [
        {
            'content': 'Machine learning is a subset of AI...',
            'metadata': {'source': 'github', 'file': 'intro.md'}
        },
        {
            'content': 'Deep learning uses neural networks...',
            'metadata': {'source': 'web', 'url': 'https://example.com'}
        },
    ]
    
    # Create and test pipeline
    pipeline = IndexingPipeline()
    pipeline.index_documents(test_docs)
    
    # Test query
    results = pipeline.query("What is machine learning?", k=2)
    
    print("\nSearch Results:")
    for metadata, score in results:
        print(f"  Score: {score:.3f}")
        print(f"  Source: {metadata['source']}")
        print()
        
        

# ============================================================
# TODO: Future Enhancements and Scalability Planning
# ============================================================

# --- 1. CORE SCALABILITY & ACCURACY (High Priority).

# TODO: OPTIMIZATION 1.1: DECOUPLE & CO-FILTERING (FILTER SCALABILITY)
"""
What: Transition from the current **Post-Filtering** pattern (over-fetching results from FAISS and filtering in Python) to a **Co-Filtering** approach. This involves integrating the metadata store with a dedicated indexing solution.
Why: The current approach does not scale. For highly selective filters (e.g., searching only 10 files out of 10,000), the system must waste time retrieving and processing thousands of irrelevant vectors from FAISS. Co-filtering is required for reliable, efficient filtering.
How: 
    1. Replace the simple `self.metadata_store: List[Dict]` with a proper, dedicated database (e.g., an in-memory SQLite table, or a specialized vector database like Qdrant/Weaviate).
    2. Build a secondary index (like an inverted index) on metadata fields (e.g., 'language', 'file_path').
    3. Modify the search logic to: **FIRST** use the metadata index to get a list of target Vector IDs, and **SECOND** instruct FAISS to *only* search within those IDs.
When: Before deployment to a large, multi-repository environment, or when filter usage becomes frequent, as the current filtering method will become the primary performance bottleneck.
"""
# TODO: OPTIMIZATION 1.2: STREAMING (MEMORY SAFETY)
"""
What: Redesign the chunking orchestrator (e.g., `chunk_all`) to accept a **generator of documents** from the Loader step and use a generator pattern (`yield`) to produce **chunks** one by one.
Why: Prevents **Out-of-Memory (OOM) errors** by avoiding the creation of a single, massive list of *all* chunks in memory. It maintains a constant, low memory footprint, making the ingestion pipeline memory-safe regardless of the final chunk count.
How: Modify the `chunk_all` method to use `yield` instead of `return` a list. The subsequent embedding and ingestion steps must be refactored to consume this generator, processing a small batch of chunks (e.g., 32-64) at a time before discarding them.
When: Before processing large datasets (tens of thousands of documents or more) where the aggregate chunk list size could exceed available RAM.
"""

# TODO: OPTIMIZATION 1.3: DISTRIBUTED EMBEDDING (INGESTION THROUGHPUT)
"""
What: Implement a **distributed processing framework** (like Ray, Spark, or a cloud batch service) to run the `model.encode` step across a cluster of multiple GPU-equipped machines.
Why: The embedding (model.encode) step is highly **CPU/GPU-bound** and is the primary bottleneck for ingestion speed. Distributing this task across a cluster allows for massive parallelization, drastically reducing the total time required to process the dataset.
How: Split the source documents into smaller files (shards). Deploy the embedding logic to multiple workers, each reading a shard and independently embedding and ingesting the results.
When: When indexing large document sets (millions/billions of chunks) where the required embedding time on a single machine is measured in days or weeks.
"""

# --- 2. CORE ROBUSTNESS & SECURITY (Medium Priority) --- 

# TODO: OPTIMIZATION 2.1: METADATA SERIALIZATION REPLACEMENT
"""
What: Replace the current `pickle` serialization of the `metadata_store` and database state with a more robust and secure format.
Why: 
    1. **Security:** Pickling from untrusted sources is a major security risk (allowing arbitrary code execution).
    2. **Portability:** Pickled files are often incompatible across different Python versions, making them fragile for long-term storage or multi-machine deployment.
    3. **Readability:** The binary format prevents easy human inspection or integration with non-Python tools.
How: Replace `pickle.dump`/`pickle.load` with a standard, human-readable format. Options include:
    * **JSON Lines (.jsonl):** Ideal for line-by-line streaming of metadata dictionaries.
    * **SQLite:** A dedicated, lightweight relational database for robust state management.
When: Before transitioning from prototyping/local development to a shared or production environment where security and long-term data management are primary concerns.
"""

# --- 3. ARCHITECTURE & OPERATIONAL COMPLEXITY (Long-term) --- 

# TODO: EXTENSION 3.1: UPDATES AND DELETIONS (DATA LIFECYCLE)
"""
What: Implement a data lifecycle management feature to handle document updates and deletions.
Why: The current FAISS index is append-only, leading to stale and unremovable data over time. A production system must support the ability to remove irrelevant information.
How: Implement a "soft delete" flag in the metadata store and a separate, scheduled job to rebuild (re-index) the FAISS index periodically, purging all soft-deleted vectors.
When: When the source data is dynamic (documents are frequently updated, edited, or removed) to prevent incorrect or irrelevant search results.
"""
# TODO: EXTENSION 3.2: SHARDING (HORIZONTAL SCALABILITY)
"""
What: Introduce a sharding mechanism to distribute the index and metadata across multiple compute nodes.
Why: The current implementation is limited by the RAM/disk space of a single machine. Sharding is required to scale the database size to hundreds of millions of vectors (hundreds of GBs).
How: Implement a cluster coordinator to manage data partitioning and route search queries to the appropriate shards (e.g., using a hash of the vector ID to determine its physical location).
When: When the index size approaches or exceeds the physical memory limits of the largest single server available.
"""
# TODO: EXTENSION 3.3: REAL-TIME INGESTION
"""
What: Refactor the ingestion pipeline to efficiently support real-time, low-latency addition of single documents.
Why: The current batch-optimized `add_documents` method is inefficient for frequent, small updates (e.g., a user saving a configuration file). Real-time capability is key for live systems.
How: Explore using a different index type or a hybrid approach optimized for fast writes, potentially by writing to a small, temporary, in-memory index that is later merged into the main index.
When: When the system needs to support live updates to the codebase/documentation that must be immediately searchable.
"""
# TODO: EXTENSION 3.4: DISTRIBUTED VECTOR DATABASE (STORAGE SCALABILITY)
"""
What: Replace the single, in-memory FAISS instance (`self.vector_db`) with a **production-grade, distributed Vector Database** solution.
Why: A single FAISS index is limited by the RAM/disk of one machine, making it impossible to store indices containing billions of vectors. A distributed system is essential for horizontal scaling.
How: Migrate the indexing logic to target a managed service (e.g., Pinecone, Weaviate, Qdrant) or a self-hosted clustered database (e.g., Milvus). This database must be capable of automatically **sharding** the index and handling simultaneous writes from multiple ingestion workers (from OPTIMIZATION 1.2).
When: Immediately, if the project is expected to scale beyond the capacity (typically 10-100 GB) of a single server's memory.
"""