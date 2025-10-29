"""
Advanced RAG Features
1. Hybrid Search (Keyword + Semantic)
2. Re-ranking
3. Query routing
4. Response validation
"""

# ============================================================
# 1. HYBRID SEARCH (BM25 + Semantic)
# ============================================================

class HybridSearch:
    """
    Combine keyword search (BM25) with semantic search
    
    Why?
    - Semantic search: Good for concept matching
    - Keyword search: Good for exact terms (function names, error messages)
    - Hybrid: Best of both!
    """
    
    def __init__(self, documents: List[str]):
        from rank_bm25 import BM25Okapi
        import nltk
        nltk.download('punkt', quiet=True)
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Store documents
        self.documents = documents
    
    def search(self, query: str, semantic_results: List[Tuple[int, float]], alpha: float = 0.5, k: int = 5):
        """
        Hybrid search combining BM25 and semantic
        
        Args:
            query: Search query
            semantic_results: List of (doc_idx, semantic_score) from vector search
            alpha: Weight for semantic vs keyword (0=keyword only, 1=semantic only)
            k: Number of results
        
        Returns:
            List of (doc_idx, combined_score)
        """
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-10)
        
        # Create semantic score dict
        semantic_dict = {idx: score for idx, score in semantic_results}
        
        # Combine scores
        combined_scores = {}
        for idx in range(len(self.documents)):
            semantic_score = semantic_dict.get(idx, 0.0)
            bm25_score = bm25_scores[idx]
            
            # Weighted combination
            combined = alpha * semantic_score + (1 - alpha) * bm25_score
            combined_scores[idx] = combined
        
        # Sort and return top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return sorted_results


# ============================================================
# 2. CROSS-ENCODER RE-RANKING
# ============================================================

class CrossEncoderReranker:
    """
    Re-rank results using cross-encoder
    
    Why?
    - Bi-encoder (sentence-transformers): Fast, separate embeddings
    - Cross-encoder: Slow, but more accurate (queries + doc together)
    
    Strategy:
    1. Use bi-encoder to get top-100 candidates (fast)
    2. Use cross-encoder to re-rank top-100 to top-10 (accurate)
    """
    
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: List[str], scores: List[float] = None, top_k: int = 5):
        """
        Re-rank documents using cross-encoder
        
        Args:
            query: Search query
            documents: List of document texts
            scores: Initial scores (optional)
            top_k: Number of results to return
        
        Returns:
            List of (doc_idx, rerank_score)
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get cross-encoder scores
        cross_scores = self.model.predict(pairs)
        
        # Combine with initial scores if provided
        if scores is not None:
            # Weighted combination
            combined = 0.7 * cross_scores + 0.3 * np.array(scores)
        else:
            combined = cross_scores
        
        # Sort and return top-k
        ranked_indices = np.argsort(combined)[::-1][:top_k]
        
        return [(idx, combined[idx]) for idx in ranked_indices]


# ============================================================
# 3. QUERY ROUTING
# ============================================================

class QueryRouter:
    """
    Route queries to appropriate handlers
    
    Query Types:
    1. Code search: "How does X function work?"
    2. Conceptual: "What is the architecture?"
    3. Debugging: "Why is X failing?"
    4. Setup: "How do I install?"
    """
    
    def __init__(self):
        # Simple keyword-based routing (can use classifier)
        self.route_patterns = {
            'code_search': ['function', 'class', 'method', 'how does', 'implementation'],
            'conceptual': ['architecture', 'design', 'overview', 'explain', 'what is'],
            'debugging': ['error', 'bug', 'failing', 'not working', 'why'],
            'setup': ['install', 'setup', 'configure', 'requirements', 'dependencies']
        }
    
    def route(self, query: str) -> str:
        """
        Determine query type
        
        Returns:
            Query type: 'code_search', 'conceptual', 'debugging', 'setup'
        """
        query_lower = query.lower()
        
        scores = {}
        for route_type, patterns in self.route_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            scores[route_type] = score
        
        # Return highest scoring route
        best_route = max(scores.items(), key=lambda x: x[1])
        
        if best_route[1] > 0:
            return best_route[0]
        else:
            return 'conceptual'  # Default
    
    def get_prompt_for_route(self, route_type: str) -> str:
        """Get specialized prompt for each route type"""
        prompts = {
            'code_search': """You are a code analysis assistant. Focus on:
- Specific function/class implementations
- Code examples
- Technical details
- File locations

Context: {context}
Question: {question}
Answer with code references:""",
            
            'conceptual': """You are a technical documentation assistant. Focus on:
- High-level architecture
- Design patterns
- Module relationships
- Best practices

Context: {context}
Question: {question}
Explain clearly:""",
            
            'debugging': """You are a debugging assistant. Focus on:
- Common error causes
- Troubleshooting steps
- Related code sections
- Known issues

Context: {context}
Question: {question}
Provide debugging guidance:""",
            
            'setup': """You are a setup/installation assistant. Focus on:
- Installation steps
- Configuration requirements
- Dependencies
- Getting started

Context: {context}
Question: {question}
Provide setup instructions:"""
        }
        
        return prompts.get(route_type, prompts['conceptual'])


# ============================================================
# 4. RESPONSE VALIDATION
# ============================================================

class ResponseValidator:
    """
    Validate RAG responses for quality
    
    Checks:
    1. Answer is grounded in context
    2. No hallucination
    3. Answers the question
    """
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def validate(self, question: str, answer: str, context: str) -> Dict:
        """
        Validate response quality
        
        Returns:
            Dict with validation results
        """
        # 1. Check if answer is grounded in context
        grounding_score = self._check_grounding(answer, context)
        
        # 2. Check if answer addresses question
        relevance_score = self._check_relevance(question, answer)
        
        # 3. Check for hedge words (uncertainty indicators)
        hedge_words = ['maybe', 'possibly', 'might', 'could be', 'not sure', "don't know"]
        has_hedges = any(word in answer.lower() for word in hedge_words)
        
        # 4. Check answer length (too short = low confidence)
        is_sufficient_length = len(answer.split()) > 10
        
        # Overall confidence
        confidence = (grounding_score + relevance_score) / 2
        
        validation = {
            'is_valid': confidence > 0.6 and is_sufficient_length,
            'confidence': confidence,
            'grounding_score': grounding_score,
            'relevance_score': relevance_score,
            'has_uncertainty': has_hedges,
            'is_sufficient_length': is_sufficient_length
        }
        
        return validation
    
    def _check_grounding(self, answer: str, context: str) -> float:
        """Check if answer is grounded in context"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        answer_emb = self.model.encode([answer])
        context_emb = self.model.encode([context])
        
        similarity = cosine_similarity(answer_emb, context_emb)[0][0]
        return float(similarity)
    
    def _check_relevance(self, question: str, answer: str) -> float:
        """Check if answer addresses question"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        question_emb = self.model.encode([question])
        answer_emb = self.model.encode([answer])
        
        similarity = cosine_similarity(question_emb, answer_emb)[0][0]
        return float(similarity)


# ============================================================
# 5. ENHANCED RAG WITH ALL FEATURES
# ============================================================

class EnhancedRAG(ConversationalRAG):
    """
    Production RAG with all advanced features
    """
    
    def __init__(self, model_name='llama2'):
        super().__init__(model_name)
        
        self.reranker = CrossEncoderReranker()
        self.router = QueryRouter()
        self.validator = ResponseValidator()
        self.hybrid_search = None
    
    def enable_hybrid_search(self, documents: List[str]):
        """Enable hybrid search"""
        self.hybrid_search = HybridSearch(documents)
        print("✅ Hybrid search enabled")
    
    def query_with_enhancements(
        self,
        question: str,
        k: int = 10,
        use_reranking: bool = True,
        use_routing: bool = True,
        validate_response: bool = True
    ):
        """
        Enhanced query with all features
        
        Pipeline:
        1. Route query to appropriate handler
        2. Retrieve candidates (semantic or hybrid)
        3. Re-rank with cross-encoder
        4. Generate answer with specialized prompt
        5. Validate response
        """
        print(f"\n{'='*70}")
        print(f"Enhanced Query: {question}")
        print(f"{'='*70}\n")
        
        # Step 1: Route query
        if use_routing:
            route = self.router.route(question)
            print(f"1. Query routed to: {route}")
            specialized_prompt = self.router.get_prompt_for_route(route)
        else:
            route = 'general'
            specialized_prompt = self.prompt_template
        
        # Step 2: Retrieve candidates
        print(f"2. Retrieving top-{k} candidates...")
        if self.vectorstore is None:
            raise ValueError("Must index documents first!")
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        
        # Step 3: Re-rank
        if use_reranking:
            print("3. Re-ranking with cross-encoder...")
            documents = [doc.page_content for doc, _ in docs_and_scores]
            scores = [score for _, score in docs_and_scores]
            
            reranked_indices = self.reranker.rerank(
                question,
                documents,
                scores,
                top_k=5
            )
            
            # Get top re-ranked docs
            final_docs = [docs_and_scores[idx][0] for idx, _ in reranked_indices]
        else:
            final_docs = [doc for doc, _ in docs_and_scores[:5]]
        
        # Step 4: Generate answer
        print("4. Generating answer...")
        context = "\n\n".join([doc.page_content for doc in final_docs])
        
        prompt = specialized_prompt.format(context=context, question=question)
        answer = self.llm(prompt)
        
        # Step 5: Validate
        if validate_response:
            print("5. Validating response...")
            validation = self.validator.validate(question, answer, context)
            print(f"   Confidence: {validation['confidence']:.2f}")
            print(f"   Grounded: {validation['grounding_score']:.2f}")
            print(f"   Relevant: {validation['relevance_score']:.2f}")
        else:
            validation = None
        
        response = {
            'question': question,
            'answer': answer,
            'route': route,
            'sources': [
                {
                    'file': doc.metadata.get('file_path', 'Unknown'),
                    'content': doc.page_content[:200] + "..."
                }
                for doc in final_docs
            ],
            'validation': validation
        }
        
        print(f"\n{'='*70}\n")
        
        return response


# ============================================================
# DEMO
# ============================================================

def demo_enhanced_features():
    """Demonstrate advanced features"""
    
    # Create enhanced RAG
    rag = EnhancedRAG()
    
    # ... (index documents first)
    
    # Query with all enhancements
    result = rag.query_with_enhancements(
        "How does the authentication function work?",
        k=10,
        use_reranking=True,
        use_routing=True,
        validate_response=True
    )
    
    print(f"Answer: {result['answer']}")
    print(f"\nRoute: {result['route']}")
    print(f"Validation: {result['validation']}")