"""
FastAPI Backend for RepoRAG
Production-ready REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    use_reranking: bool = True
    filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    confidence: Optional[float] = None
    processing_time_ms: float

class IndexRequest(BaseModel):
    repo_path: Optional[str] = None
    urls: Optional[List[str]] = None
    videos: Optional[List[str]] = None

class IndexResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    total_documents: int


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RepoRAG API",
    description="Ask questions about any GitHub repository",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system
rag_system = None


# ============================================================
# ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    print("Initializing RepoRAG system...")
    rag_system = EnhancedRAG()
    
    # Try to load existing index
    try:
        rag_system.load('./reporag_index')
        print("✅ Loaded existing index")
    except:
        print("No existing index found")

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "RepoRAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=rag_system is not None,
        index_loaded=rag_system.vectorstore is not None if rag_system else False,
        total_documents=rag_system.vectorstore.index.ntotal if rag_system and rag_system.vectorstore else 0
    )

@app.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest):
    """
    Index a repository
    
    Example:
```
    POST /index
    {
        "repo_path": "/path/to/repo",
        "urls": ["https://docs.example.com"],
        "videos": ["https://youtube.com/watch?v=xxx"]
    }
```
    """
    import time
    start_time = time.time()
    
    try:
        # Create processor
        processor = DocumentProcessor()
        
        # Build config
        config = {}
        if request.repo_path:
            config['github_repos'] = [request.repo_path]
        if request.urls:
            config['urls'] = request.urls
        if request.videos:
            config['videos'] = request.videos
        
        if not config:
            raise HTTPException(status_code=400, detail="Must provide at least one source")
        
        # Process documents
        documents = processor.process_all(config)
        
        # Chunk
        chunker = CodeAwareChunker()
        chunks = chunker.chunk_all(documents)
        
        # Index
        rag_system.index_documents(chunks)
        
        # Save
        rag_system.save('./reporag_index')
        
        processing_time = (time.time() - start_time) * 1000
        
        return IndexResponse(
            status="success",
            documents_indexed=len(documents),
            chunks_created=len(chunks),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """
    Query the indexed repository
    
    "question": "How does authentication work?",
        "k": 5,
        "use_reranking": true
    }
````
    """
    import time
    start_time = time.time()
    
    try:
        if rag_system.vectorstore is None:
            raise HTTPException(
                status_code=400,
                detail="No index loaded. Please index a repository first using /index endpoint"
            )
        
        # Query with enhancements
        result = rag_system.query_with_enhancements(
            question=request.question,
            k=request.k,
            use_reranking=request.use_reranking,
            use_routing=True,
            validate_response=True
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['validation']['confidence'] if result['validation'] else None,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=QueryResponse)
async def chat_with_repository(request: QueryRequest):
    """
    Conversational query (maintains chat history)
    
    Example:
````
    POST /chat
    {
        "question": "Tell me about the main modules"
    }
    
    POST /chat
    {
        "question": "What about the first one you mentioned?"
    }
````
    """
    import time
    start_time = time.time()
    
    try:
        if rag_system.vectorstore is None:
            raise HTTPException(status_code=400, detail="No index loaded")
        
        result = rag_system.chat(request.question)
        
        processing_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=result['question'],
            answer=result['answer'],
            sources=result['sources'],
            confidence=None,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/reset")
async def reset_chat():
    """Reset conversation history"""
    try:
        rag_system.reset_conversation()
        return {"status": "success", "message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=Dict)
async def get_statistics():
    """Get index statistics"""
    try:
        if rag_system.vectorstore is None:
            return {
                "indexed": False,
                "total_documents": 0
            }
        
        return {
            "indexed": True,
            "total_documents": rag_system.vectorstore.index.ntotal,
            "dimension": rag_system.vectorstore.index.d
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
````

**Test the API:**
````bash
# Terminal 1: Start server
python api.py

# Terminal 2: Test endpoints
# Health check
curl http://localhost:8000/health

# Index a repository
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "./your-repo",
    "urls": ["https://docs.example.com"]
  }'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main architecture?",
    "k": 5
  }'
````

---

### Hour 8-9: Frontend & Documentation
````python
"""
requirements.txt - All dependencies
"""
# Core
sentence-transformers==2.2.2
faiss-cpu==1.7.4
langchain==0.1.0
numpy==1.24.3

# LLM
ollama==0.1.6

# Document Processing
beautifulsoup4==4.12.2
requests==2.31.0
youtube-transcript-api==0.6.1
rank-bm25==0.2.2

# API
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1
nltk==3.8.1
````

**README.md** - Professional documentation:
````markdown
# RepoRAG 🚀

> Ask questions about any GitHub repository using AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)

RepoRAG helps developers understand unfamiliar codebases 10x faster by enabling natural language conversations with code repositories.

## ✨ Features

- 🔍 **Multi-Source Ingestion**: GitHub repos, wikis, documentation URLs, YouTube tutorials
- 🧠 **Semantic Search**: Find relevant code by meaning, not just keywords
- 💬 **Conversational AI**: Ask follow-up questions with context
- 🎯 **Code-Aware Chunking**: Preserves function/class boundaries
- 🔄 **Hybrid Search**: Combines semantic + keyword search
- 📊 **Re-ranking**: Cross-encoder for highest accuracy
- 🚀 **Production Ready**: FastAPI backend, Docker support
- 💰 **100% Free**: Uses local LLMs (Ollama), no API costs

## 🎥 Demo
```bash
# Index a repository
curl -X POST http://localhost:8000/index -d '{
  "repo_path": "./langchain"
}'

# Ask questions
curl -X POST http://localhost:8000/query -d '{
  "question": "How does the RetrievalQA chain work?"
}'
```

## 🏗️ Architecture
````
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Query Router & Processor                    │
│  • Routes to specialized handlers                        │
│  • Optimizes query for search                           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Hybrid Search (BM25 + Vector)              │
│  • Retrieves top-100 candidates                         │
│  • Combines keyword + semantic                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Cross-Encoder Re-ranking                    │
│  • Re-ranks to top-10                                   │
│  • Higher accuracy                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              LLM Generation (Ollama)                     │
│  • Specialized prompts per query type                   │
│  • Grounded in retrieved context                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Response Validation                         │
│  • Checks grounding in context                          │
│  • Validates relevance                                  │
└─────────────────────────────────────────────────────────┘