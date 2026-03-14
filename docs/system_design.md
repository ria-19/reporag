# ============================================================
# TODO: Future Enhancements and Scalability Planning
# ============================================================

# --- 1. CORE SCALABILITY & PERFORMANCE (High Priority) ---

# TODO: OPTIMIZATION 1.1: STREAMING (MEMORY SAFETY)
"""
What: Redesign the chunking orchestrator (e.g., `chunk_all`) to accept a **generator of documents** from the Loader step and use a generator pattern (`yield`) to produce **chunks** one by one.
Why: Prevents **Out-of-Memory (OOM) errors** by avoiding the creation of a single, massive list of *all* chunks in memory. It maintains a constant, low memory footprint, making the ingestion pipeline memory-safe regardless of the final chunk count.
When: Before processing large datasets (tens of thousands of documents or more) where the aggregate chunk list size could exceed available RAM.
"""
# TODO: OPTIMIZATION 1.2: PARALLELISM (INGESTION THROUGHPUT)
"""
What: Implement Python's `concurrent.futures.ProcessPoolExecutor` in the chunking orchestrator to run the `chunk_document` function concurrently across all available CPU cores.
Why: Chunking is an **embarrassingly parallel** and often **CPU-bound** task (especially with advanced parsers). Utilizing all CPU cores drastically reduces the total wall-clock time required to process the entire document set, maximizing ingestion throughput.
When: Immediately, as the current synchronous processing is a major bottleneck for any medium-to-large codebase.
"""
# TODO: OPTIMIZATION 1.3: INCREMENTAL PROCESSING (STATE MANAGEMENT)
"""
What: Introduce a persistence layer to store the **Content Hash** (e.g., SHA-256) of a document's content after successful chunking. The chunking orchestrator will only call `chunk_document()` if the document's current hash is different from the stored hash.
Why: Enables highly efficient **incremental updates** specifically for the CPU-intensive chunking step. It saves significant CPU resources and time by preventing the re-execution of the complex chunking logic on unchanged documents.
When: High Priority. Implement once the system starts seeing high data volume or is put on a regular update schedule. 
"""

# --- 2. RESILIENCE & CODE QUALITY ---

# TODO: QUALITY 2.1: ROBUST ERROR ISOLATION
"""
What: In the chunking orchestrator, wrap the call to `chunk_document(doc)` in a dedicated `try...except` block.
Why: Implements **fault tolerance**. A failure to parse/chunk one malformed document will not cause a **cascading failure** that halts the chunking of all other documents in the queue.
When: In v1 improvement (fundamental for pipeline stability).
"""
# TODO: QUALITY 2.2: REFINE ERROR RECOVERY (Dead Letter Queue)
"""
What: Enhance the error handling (2.1) to serialize the failed document and its full error traceback, writing it to a designated storage location (a "dead letter queue" directory or file).
Why: Allows engineers to later inspect and reprocess specific documents that failed due to temporary or fixable issues, ensuring **zero data loss** without blocking the main pipeline.
When: Implement as part of the v2 production pipeline readiness.
"""

# --- 3. DATA QUALITY & EXTENSIBILITY ---

# TODO: DATA QUALITY 3.1: LANGUAGE-AWARE CONTEXT EXTRACTION (ROBUSTNESS)
"""
What: Replace the simple regex in `add_context_header` with dedicated language parsers (e.g., `tree-sitter` or Python's built-in `ast` module) to build an Abstract Syntax Tree (AST).
Why: An AST is infinitely more robust and accurate than brittle regex. It guarantees correct extraction of function, class, and method signatures, regardless of multi-line definitions or decorators.
When: Before scaling to complex, production codebases with non-trivial syntax.
"""

# TODO: EXTENSIBILITY 3.2: REFACTOR FOR STRATEGY PATTERN (MAINTAINABILITY)
"""
What: Refactor the language-specific context extraction logic into separate classes (e.g., `PythonContextAdder`, `JSContextAdder`) based on the Strategy Design Pattern.
Why: Prevents the `add_context_header` method from becoming an unmaintainable `if/elif/else` nightmare. It cleanly isolates language-specific complexity, making the pipeline easy to extend (add a new language) and maintain.
When: Immediately after implementing the AST-based parsers (3.1).
"""
# TODO: OPTIMIZATION 3.3: NON-ADJACENT CHUNK MERGING (ADVANCED SEMANTIC ACCURACY)
"""
What: Introduce a check that compares the new sentence's embedding not just to the current running chunk, but also to the embeddings of ALL previously finalized chunks. If a strong match is found with an older chunk, the algorithm could potentially merge the new sentence/section with that non-adjacent, similar chunk (e.g., merging Chunk A2 back into Chunk A1).
Why: Overcomes the limitation of the current **"greedy" algorithm** (which only looks backward one step). This preserves **global semantic context** by handling complex document structures (A-B-A, where Topic A is revisited), resulting in larger, more cohesive, and semantically pure final chunks. 
When: Low Priority/Phase 2. Implement after all performance and basic robustness issues are solved, as this method is **highly computationally expensive** due to the need for multiple cosine similarity calculations per sentence.
"""
# --- 4. CONFIGURATION MANAGEMENT (Maintainability) ---

# TODO: MAINTENANCE 4.1: EXTERNALIZE CHUNKING PARAMETERS
"""
What: Move all chunking-related parameters (e.g., `similarity_threshold`, `max_chunk_size`) from being hard-coded constants to being loaded from an external configuration source (e.g., YAML or Environment Variables).
Why: Decouples behavior from source code. This is essential for rapid experimentation (A/B testing) and tuning production performance without requiring a code change or redeployment.
"""

# ============================================================
# METRICS FOR EVALUATION
# ============================================================

# METRIC: CHUNKING THROUGHPUT
# Formula: Total documents chunked / Time taken (e.g., documents per second)
# Goal: Track the raw processing speed. Directly tied to Parallelism (1.2) fix.

# METRIC: MEMORY FOOTPRINT (PEAK)
# Formula: Maximum memory usage (GB) during a full ingestion run.
# Goal: Track efficiency and stability. Should remain constant regardless of dataset size (related to Streaming 1.1).


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