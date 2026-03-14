"""
SimpleRAG: Learn RAG from First Principles
==========================================

Build a complete RAG system in ~100 lines to understand:
- How documents become searchable (embeddings)
- How semantic search actually works (vector similarity)
- How LLMs use retrieved context (prompt engineering)

No magic. No abstractions. Just the core concepts.

Requirements:
    pip install sentence-transformers numpy requests

Setup (for local LLM):
    Terminal 1: ollama serve
    Terminal 2: ollama pull llama2
"""

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple


# ==============================================================================
# STEP 1: REPRESENT DOCUMENTS
# ==============================================================================

class Document:
    """
    A document is just content + metadata.
    Metadata helps us organize and filter documents.
    """
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata


# ==============================================================================
# STEP 2: BUILD THE RAG SYSTEM
# ==============================================================================

class SimpleRAG:
    """
    A minimal RAG implementation with four key components:
    
    1. STORAGE: Keep documents and their vector embeddings
    2. INDEXING: Convert text to semantic vectors
    3. RETRIEVAL: Find documents similar to a query
    4. GENERATION: Use retrieved docs as context for LLM
    """

    def __init__(self):
        print("🚀 Initializing SimpleRAG...")
        
        # Load a small, fast embedding model (384-dimensional vectors)
        # This model understands semantic meaning, not just keywords
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Our in-memory database
        self.documents: List[Document] = []        # Stores actual documents
        self.embeddings: np.ndarray = None         # Stores their vector representations
        
        print("✓ Ready to process documents!\n")

    # --------------------------------------------------------------------------
    # INDEXING: Add documents to our knowledge base
    # --------------------------------------------------------------------------
    
    def add_document(self, content: str, metadata: dict):
        """
        Store a document and create its semantic embedding.
        
        The embedding captures the *meaning* of the text as a vector.
        Similar documents will have similar vectors (close in vector space).
        """
        # Step 1: Store the document
        doc = Document(content, metadata)
        self.documents.append(doc)
        
        # Step 2: Convert text → vector (this is where the magic happens)
        embedding = self.embedding_model.encode(content)
        
        # Step 3: Add to our embedding matrix
        if self.embeddings is None:
            # First document: create matrix with shape (1, 384)
            self.embeddings = embedding.reshape(1, -1)
        else:
            # Stack new embedding below existing ones
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        print(f"✓ Indexed: {metadata.get('title', 'Untitled')}")

    # --------------------------------------------------------------------------
    # RETRIEVAL: Find relevant documents using semantic search
    # --------------------------------------------------------------------------
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Semantic search: Find documents by meaning, not just keywords.
        
        How it works:
        1. Convert query to a vector (same space as documents)
        2. Measure similarity between query vector and each document vector
        3. Return the k most similar documents
        
        Cosine similarity tells us how "aligned" two vectors are (0 to 1).
        """
        if len(self.documents) == 0:
            return []
        
        # Embed the query into the same vector space as documents
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarity with every document
        similarities = []
        for doc_embedding in self.embeddings:
            # Cosine similarity formula: (A·B) / (|A| × |B|)
            # Measures angle between vectors, not distance
            dot_product = np.dot(query_embedding, doc_embedding)
            magnitude_product = np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            similarity = dot_product / magnitude_product
            similarities.append(similarity)
        
        # Get indices of top-k most similar documents
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return [(self.documents[i], similarities[i]) for i in top_k_indices]

    # --------------------------------------------------------------------------
    # GENERATION: Create answer using LLM with retrieved context
    # --------------------------------------------------------------------------
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Call a local LLM (via Ollama) to generate an answer.
        
        The LLM sees both the query AND the retrieved context.
        This grounds the answer in your actual documents.
        """
        # Craft a prompt that includes retrieved context
        prompt = f"""
        Answer the question based on the context below.
        If the context doesn't contain the answer, say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 256}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "❌ Error: Could not generate answer"
                
        except Exception as e:
            return f"❌ Error calling LLM: {str(e)}"

    # --------------------------------------------------------------------------
    # PIPELINE: The complete RAG flow
    # --------------------------------------------------------------------------
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        The full RAG pipeline in three steps:
        
        1. RETRIEVE: Find k most relevant documents
        2. AUGMENT: Combine them into context
        3. GENERATE: Use context to answer the question
        """
        print(f"\n❓ Query: {question}")
        
        # Step 1: Retrieve relevant documents
        print(f"   🔍 Searching for relevant documents...")
        results = self.search(question, k=k)
        
        if not results:
            return {
                "question": question,
                "answer": "No relevant documents found.",
                "sources": [],
                "context": []
            }
        
        print(f"   📚 Found {len(results)} relevant documents:")
        
        # Step 2: Build context from retrieved documents
        context_parts = []
        sources = []
        for doc, score in results:
            title = doc.metadata.get('title', 'Untitled')
            context_parts.append(f"[{title}]\n{doc.content.strip()}")
            sources.append(title)
            print(f"      • {title} (similarity: {score:.3f})")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer using context
        print(f"   🤖 Generating answer with LLM...")
        answer = self.generate_answer(question, context)
        
        print(f"   ✓ Done!\n")
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context_parts
        }


# ==============================================================================
# DEMO: Build and query a knowledge base
# ==============================================================================

def main():
    print("=" * 70)
    print("SimpleRAG: Understanding Retrieval-Augmented Generation")
    print("=" * 70)
    print()
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Build a small knowledge base
    print("📖 Building knowledge base...\n")
    
    rag.add_document(
        """Machine learning is a subset of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed. 
        It uses algorithms that improve automatically through experience.""",
        {"title": "ML Basics", "category": "intro"}
    )
    
    rag.add_document(
        """Deep learning is a form of machine learning that uses neural networks 
        with multiple layers. It excels at processing images, text, and audio 
        by learning hierarchical representations of data.""",
        {"title": "Deep Learning", "category": "advanced"}
    )
    
    rag.add_document(
        """RAG (Retrieval-Augmented Generation) combines information retrieval 
        with text generation. It retrieves relevant documents and uses them 
        as context for generating accurate, grounded answers.""",
        {"title": "RAG Explained", "category": "techniques"}
    )
    
    rag.add_document(
        """Python is the most popular language for machine learning. 
        It provides excellent libraries like scikit-learn, PyTorch, and 
        TensorFlow, along with a simple syntax and strong community.""",
        {"title": "Python for ML", "category": "tools"}
    )
    
    # Test the system with queries
    print("\n" + "=" * 70)
    print("🧪 Testing SimpleRAG")
    print("=" * 70)
    
    test_queries = [
        "What is machine learning?",
        "Tell me about RAG",
        "Which programming language is best for ML?"
    ]
    
    for q in test_queries:
        result = rag.query(q, k=2)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("\n" + "-" * 70 + "\n")


if __name__ == "__main__":
    main()


# ==============================================================================
# KEY INSIGHTS
# ==============================================================================
"""
What makes RAG work?

1. SEMANTIC EMBEDDINGS
   - Text → vectors that capture meaning
   - Similar meanings = similar vectors
   - This enables "understanding" instead of keyword matching

2. SIMILARITY SEARCH
   - Query gets embedded into the same vector space
   - Cosine similarity finds documents with related meaning
   - Fast and effective for finding relevant context

3. CONTEXT AUGMENTATION
   - Retrieved documents become part of the LLM prompt
   - Grounds the answer in your specific knowledge base
   - Reduces hallucinations and improves accuracy

4. SIMPLICITY
   - Core RAG needs: embeddings + similarity + LLM call
   - No complex frameworks required to understand the fundamentals
   - Production systems add optimizations, but the core stays the same
"""