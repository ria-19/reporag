"""
Why Chunking Matters

Scenario: 100-page technical manual

Option 1: Embed entire document (1 vector)
- Problem: Too broad, loses specificity
- Query "How do I reset password?" → Matches entire manual

Option 2: Embed each sentence (1000s of vectors)
- Problem: Too granular, loses context
- Sentence "Click the button" → No context which button

Option 3: Chunk into ~500-token passages
- Sweet spot: Enough context, still specific
- Passage contains full explanation with context

This is THE most important decision in RAG
Good Chunk
- Semantic Cohesion: All info within chunk roughly for one/same topic
- Self-Contained: Can be understood on its own
- Appropriate Size: Context Window + Context handling (no Context fragmentation + no Semantic dilution)
- Respects Boundaries: Logical + Semantic

"""

# Experiment with different chunk sizes

document = """
Machine learning is a subset of artificial intelligence that focuses on 
learning from data. It enables computers to learn and improve from experience 
without being explicitly programmed.

There are three main types of machine learning:
1. Supervised learning uses labeled data
2. Unsupervised learning finds patterns in unlabeled data  
3. Reinforcement learning learns through trial and error

Deep learning is a subset of machine learning that uses neural networks 
with multiple layers. It has revolutionized computer vision and natural 
language processing tasks.
"""

# Chunk Strategy 1: Fixed character length (naive)
def chunk_fixed(text, chunk_size=100):
    """ Split every N characters"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# Chunk Strategy 2: Sentence-aware (better)
def chunk_sentences(text, target_length=150):
    """ Split on sentences, combine to reach target length"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence + current_chunk) < target_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
             chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Chunk Strategy 3: Semantic-aware (best)
def chunk_recursive(text, chunk_size=200, overlap=20):
    """
    Recursive splitting with overlap
    - Try paragraph boundaries first
    - Then sentence boundaries
    - Then word boundaries
    - Finally character boundaries
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text, separators):
        if len(text) <= chunk_size:
            return [text]
        
        # Try each separator
        for sep in separators:
            if sep in text:
                splits = text.split(sep)
                chunks = []
                current = ""
                
                for split in splits:
                    if len(current + sep + split) <= chunk_size:
                        current += sep + split if current else split
                    else:
                        if current:
                            chunks.append(current)
                        current = split
                
                if current:
                    chunks.append(current)
                
                return chunks
        
        # Fallback: split at chunk_size
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Add overlap
    chunks = split_text(text, separators)
    overlapping_chunks = []
    
    for i, chunk in enumerate(chunks):
        if i > 0:
            # Add overlap from previous chunk
            prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            chunk = prev_overlap + " " + chunk
        overlapping_chunks.append(chunk)
    
    return overlapping_chunks


# Compare strategies
print("Strategy 1 - Fixed length:")
chunks1 = chunk_fixed(document, 100)
for i, chunk in enumerate(chunks1[:3]):
    print(f"  Chunk {i}: {chunk[:60]}...")

print("\nStrategy 2 - Sentence-aware:")
chunks2 = chunk_sentences(document, 150)
for i, chunk in enumerate(chunks2[:3]):
    print(f"  Chunk {i}: {chunk[:60]}...")

print("\nStrategy 3 - Recursive with overlap:")
chunks3 = chunk_recursive(document, 200, 20)
for i, chunk in enumerate(chunks3[:3]):
    print(f"  Chunk {i}: {chunk[:60]}...")

# KEY INSIGHT:
# - Chunk size: 500-1000 tokens (not characters!)
# - Overlap: 10-20% of chunk size
# - Respect semantic boundaries (paragraphs, sentences)



# Understand
# - Tokens: The fundamental unit of meaning for a Language Model. A token can be a whole word, a part of a word (a subword), or even just a single character or punctuation. 
# -  Models don't "read" characters; they "see" tokens.

"""
EXPERIMENT: Does chunking strategy matter?

Test with a real question-answering scenario
"""

# Full document
long_doc = """
The company's remote work policy allows employees to work from home 
up to 3 days per week. Core hours are 10am to 3pm in local timezone.
All meetings should be recorded for asynchronous team members.

The PTO policy includes 20 vacation days, 10 sick days, and 5 personal days.
After 2 years of tenure, the company switches to an unlimited PTO policy.

Equipment provided includes a laptop (MacBook Pro or equivalent), 
$500 annual stipend for home office setup, and $100 monthly for internet.
"""

# Query
query = "What is the remote work policy?"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

 
# Test different chunking strategies
model = SentenceTransformer('all-MiniLM-L6-v2')


def test_chunking_strategy(chunks, query):
    """See which chunks are retrieved"""
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])[0]
    
    similarities = [
        cosine_similarity([query_embedding], [chunk_emb])[0][0]
        for chunk_emb in chunk_embeddings
    ]
    
    best_idx = np.argmax(similarities)
    return chunks[best_idx], similarities[best_idx]

# Strategy 1: Whole document (no chunking)
chunks_none = [long_doc]
result1, score1 = test_chunking_strategy(chunks_none, query)
print(f"No chunking - Score: {score1:.3f}")
print(f"Retrieved: {result1[:100]}...\n")

# Strategy 2: Paragraph-level chunks
chunks_para = long_doc.split('\n\n')
result2, score2 = test_chunking_strategy(chunks_para, query)
print(f"Paragraph chunks - Score: {score2:.3f}")
print(f"Retrieved: {result2[:100]}...\n")

# Strategy 3: Sentence-level chunks
chunks_sent = [s.strip() for s in long_doc.split('. ') if s.strip()]
result3, score3 = test_chunking_strategy(chunks_sent, query)
print(f"Sentence chunks - Score: {score3:.3f}")
print(f"Retrieved: {result3[:100]}...\n")

# OBSERVE:
# - Paragraph chunks usually work best
# - They have enough context
# - But stay focused on one topic

# PRACTICAL TAKEAWAY:
# For RAG: Use RecursiveCharacterTextSplitter with:
# - chunk_size = 500 tokens (~400 words)
# - chunk_overlap = 50 tokens
# - separators = ["\n\n", "\n", ". ", " "]

