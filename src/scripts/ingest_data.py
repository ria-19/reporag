"""
Smart Chunking for Code
Understanding: Code structure matters for retrieval
"""

from typing import List, Dict
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

# ============================================================
# 1. CODE-AWARE CHUNKING
# ============================================================

class CodeAwareChunker:
    """
    Chunk code while preserving structure
    
    Key Insights:
    - Don't split functions in middle
    - Keep imports with related code
    - Preserve class structure
    - Add context headers
    """
    
    def __init__(self):
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
        if 'language' in metadata and metadata['language'] == 'python':
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
        self.similarity_threshold = 0.7
    
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
                # Update embedding (average)
                current_embedding = (current_embedding + embeddings[i]) / 2
        
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