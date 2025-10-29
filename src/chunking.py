"""
Smart Chunking for Code
Goal: Transform large, clean documents into small, optimized pieces of context that are both semantically coherent 
and sized perfectly for both the vector database and LLM.
"""

from typing import List, Dict
from langchain_text_splitters import (RecursiveCharacterTextSplitter, Language)

# ============================================================
# 1. CODE-AWARE CHUNKING
# ============================================================

class CodeAwareChunker:  # For code, the atomic unit of meaning is not a paragraph, but a logical block like a function or a class.
    """
    Chunk code while preserving structure
    
    Key Insights:
    - Don't split functions in middle
    - Keep imports with related code 
    - Preserve class structure
    - Add context headers
    """
    
    def __init__(self):  # Strategy Pattern
        # Different splitters for different languages
        self.splitters = {
            'python': RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'javascript': RecursiveCharacterTextSplitter.from_language(
                language=Language.JS,
                chunk_size=1000,
                chunk_overlap=200
            ),
            'markdown': RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200
            ),
        }
        
        # Fallback for unknown languages
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    # Enrichment
    def add_context_header(self, chunk: str, metadata: Dict) -> str: 
        """
        Add context to chunk for better retrieval
        
        Example:
        ```python
            # File: src/utils/helpers.py
            # Function: calculate_similarity
            
            def calculate_similarity(vec1, vec2):
                ...
        ```
        """
        header_parts = []
        
        # File path
        if 'file_path' in metadata:
            header_parts.append(f"File: {metadata['file_path']}")
        
        # Function/class name (extract from chunk)
        if 'language' in metadata and metadata['language'] == 'python': # heuristic filter
            # Extract function/class names
            import re
            func_match = re.search(r'def\s+(\w+)', chunk)
            class_match = re.search(r'class\s+(\w+)', chunk)
            
            if func_match:
                header_parts.append(f"Function: {func_match.group(1)}")
            if class_match:
                header_parts.append(f"Class: {class_match.group(1)}")
        
        if header_parts:
            header = '\n'.join(header_parts) + '\n\n'
            # A form of Metadata Indexing (injecting the most important metadata directly into text to be embedded.
            return header + chunk 
        
        return chunk
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk single document with context
        """
        content = document['content']
        metadata = document['metadata']
        language = metadata.get('language', 'text')
        
        # Get appropriate splitter
        splitter = self.splitters.get(language, self.default_splitter)
        
        # Split content
        chunks = splitter.split_text(content)
        
        # Add context and metadata to each chunk
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            # Add context header
            chunk_with_context = self.add_context_header(chunk, metadata)
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            chunked_docs.append({
                'content': chunk_with_context,
                'metadata': chunk_metadata
            })
        
        return chunked_docs
    
    def chunk_all(self, documents: List[Dict]) -> List[Dict]:
        """Chunk all documents"""
        all_chunks = []
        
        print(f"Chunking {len(documents)} documents...")
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        return all_chunks


# ============================================================
# 2. SEMANTIC CHUNKING (Advanced)
# ============================================================

class SemanticChunker:
    """
    Chunk based on semantic similarity
    
    Idea: Split when topic changes significantly
    More intelligent than fixed-size chunking
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.25
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_by_similarity(self, text: str, max_chunk_size=500) -> List[str]:
        """
        Chunk based on semantic similarity
        
        Algorithm:
        1. Split into sentences
        2. Embed each sentence
        3. Start new chunk when similarity drops
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Embed all sentences
        embeddings = self.model.encode(sentences)
        
        # Group into chunks
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        sentence_count = 1
        
        for i in range(1, len(sentences)):
            # Compute similarity with current chunk
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                [current_embedding],
                [embeddings[i]]
            )[0][0]
            
            # Check if should start new chunk
            current_length = sum(len(s) for s in current_chunk)
            
            if similarity < self.similarity_threshold or current_length > max_chunk_size:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
            else:
                # Add to current chunk
                current_chunk.append(sentences[i])
                new_sentence_count = sentence_count + 1
                # Calculate the weighted sum and divide by the new count (N+1)
                weighted_sum = (current_embedding * sentence_count) + embeddings[i]
                # Update embedding (average)
                current_embedding = weighted_sum / new_sentence_count
                # Update the count for the next iteration
                sentence_count = new_sentence_count
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# ============================================================
# 3. COMPARISON: FIXED VS SEMANTIC CHUNKING
# ============================================================

def compare_chunking_strategies(text: str):
    """Compare different chunking approaches"""
    
    print("Testing chunking strategies...")
    print(f"Input text length: {len(text)} characters\n")
    
    # Strategy 1: Fixed size
    fixed_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    fixed_chunks = fixed_splitter.split_text(text)
    
    print(f"Fixed-size chunking: {len(fixed_chunks)} chunks")
    for i, chunk in enumerate(fixed_chunks[:2]):
        print(f"  Chunk {i}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:100]}...")
    
    # Strategy 2: Semantic
    semantic_chunker = SemanticChunker()
    semantic_chunks = semantic_chunker.chunk_by_similarity(text, max_chunk_size=500)
    
    print(f"\nSemantic chunking: {len(semantic_chunks)} chunks")
    for i, chunk in enumerate(semantic_chunks[:2]):
        print(f"  Chunk {i}: {len(chunk)} chars")
        print(f"    Preview: {chunk[:100]}...")
    
    return fixed_chunks, semantic_chunks


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    # Test with sample code
    sample_code = """
    import numpy as np
    from typing import List

    class VectorDatabase:
        def __init__(self, dimension: int):
            self.dimension = dimension
            self.vectors = []
            self.metadata = []
        
        def add(self, vector: np.ndarray, metadata: dict):
            '''Add vector to database'''
            if len(vector) != self.dimension:
                raise ValueError("Wrong dimension")
            self.vectors.append(vector)
            self.metadata.append(metadata)
        
        def search(self, query: np.ndarray, k: int = 5):
            '''Search for similar vectors'''
            # Compute similarities
            similarities = []
            for vec in self.vectors:
                sim = np.dot(query, vec)
                similarities.append(sim)
            
            # Return top-k
            indices = np.argsort(similarities)[-k:][::-1]
            return indices
    """
    
    # Test chunking
    chunker = CodeAwareChunker()
    doc = {
        'content': sample_code,
        'metadata': {
            'source': 'github',
            'file_path': 'src/database.py',
            'language': 'python'
        }
    }
    
    chunks = chunker.chunk_document(doc)
    
    print("Chunking Results:")
    for chunk in chunks:
        print(f"\nChunk {chunk['metadata']['chunk_id']}:")
        print(chunk['content'][:200])
        print("...")
        
        
        
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
