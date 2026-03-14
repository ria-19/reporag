"""
EMBEDDINGS: The Core Concept

Problem: Computers don't understand "dog" and "puppy" are similar
Solution: Represent words/sentences as vectors in high-dimensional space

Key Insight: Similar meaning → Similar vectors (close in space)
"""

# 1. Toy Example: Manual Embeddings
# Let' s create our own simple embedding space

words = ["dog", "puppy", "cat", "kitten", "car", "vehicle"]

# 2D embedding space (x=animal, y=young)
manual_embeddings = {
    "dog":     [1.0, 0.0],   # animal, not young
    "puppy":   [1.0, 1.0],   # animal, young
    "cat":     [0.9, 0.0],   # animal, not young
    "kitten":  [0.9, 1.0],   # animal, young
    "car":     [0.0, 0.0],   # not animal, not young
    "vehicle": [0.0, 0.0],   # not animal, not young
}

# Intuition: Close vectors = similar meaning
import numpy as np
import matplotlib.pyplot as plt
EPSILON = 1e-8

def plot_embeddings(embeddings):
    """Visualize our toy embedding space"""
    plt.figure(figsize=(10, 8))
    
    for word, [x, y] in embeddings.items():
        plt.scatter(x, y, s=200)
        plt.annotate(word, (x, y), fontsize=14)
    
    plt.xlabel('Animal-ness')
    plt.ylabel('Youth-ness')
    plt.title('Toy 2D Embedding Space')
    plt.grid(True)
    plt.show()

plot_embeddings(manual_embeddings)

# Observe: dog and puppy are close, car is far from animals

"""
EXERCISE 1: Implement Similarity from Scratch
"""

def cosine_similarity(vec1, vec2):
    """
    Measure similarity between two vectors
    
    Intuition: Angle between vectors
    - 0° (same direction) = similarity 1.0
    - 90° (perpendicular) = similarity 0.0
    - 180° (opposite) = similarity -1.0
    
    Formula: cos(θ) = (A·B) / (|A||B|)  
    Meaning: How much the vectors are aligned / magnitude (length) to focus only on direction
    """
    
    # Dot product (how aligned are vectors?)
    dot_product = np.dot(vec1, vec2)
    
    # Magnitudes (length of vectors)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Add epsilon to the denominator to prevent division by zero
    denominator = (magnitude1 * magnitude2) + EPSILON
    
    # Calculate similarity
    similarity = dot_product / denominator
    
    return similarity

    
# Test it
dog_vec = manual_embeddings["dog"]
puppy_vec = manual_embeddings["puppy"]
car_vec = manual_embeddings["car"]

print(f"dog ↔ puppy: {cosine_similarity(dog_vec, puppy_vec):.3f}")  # High
print(f"dog ↔ car:   {cosine_similarity(dog_vec, car_vec):.3f}")    # Low
print(f"dog ↔ cat:   {cosine_similarity(dog_vec, manual_embeddings['cat']):.3f}")  # Medium-High

# Why this matters: This is EXACTLY what RAG does, in hundreds or even thousands of dimensions

"""
EXERCISE 2: Explore Real Embeddings
"""
from sentence_transformers import SentenceTransformer

# Load model (this is what we'll use in RAG)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
sentences = [
    "The dog is barking",
    "A puppy is playing",
    "The cat is sleeping",
    "A car is driving",
    "Machine learning is fascinating"
]

embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings.shape}")  # (5, 384) {rows(sentences), cols(dimensions of each)}
print(f"Each sentence becomes 384 numbers")

# Compute all pairwise similarities
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings) # 5 * 5 matrix here 

# Visualize
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity_matrix, 
    annot=True, 
    fmt='.2f',
    xticklabels=sentences,
    yticklabels=sentences,
    cmap='coolwarm'
)
plt.title('Semantic Similarity Matrix')
plt.tight_layout()
plt.show()

# OBSERVE:
# - Dog and puppy sentences are similar (~0.6-0.7)
# - Dog and car sentences are dissimilar (~0.2)
# - This is WITHOUT sharing any words!

# KEY INSIGHT: Embeddings capture MEANING, not just keywords


"""
UNDERSTANDING: How does the model learn embeddings?

Training Process (simplified):
1. Start with random embeddings
2. Show model millions of sentence pairs
3. If sentences are similar → push embeddings closer
4. If sentences are different → push embeddings apart
5. Repeat until embeddings capture semantic meaning

Example Training Data:
- "The dog ran" | "A puppy is running" → SIMILAR (push closer)
- "The dog ran" | "Stock market fell" → DIFFERENT (push apart)

This is called Contrastive Learning
"""


sentence = "What is machine learning?"
embedding = model.encode(sentence)

print(f"Sentence: '{sentence}'")
print(f"Becomes: Vector of {len(embedding)} numbers")
print(f"First 10 dimensions: {embedding[:10]}")
print(f"Each dimension captures some semantic aspect")

# Why 384 dimensions?
# - More dimensions = more nuance
# - Fewer dimensions = faster but less precise
# - 384 is sweet spot for balance

# PRACTICAL INSIGHT:
# You don't need to understand HOW the model learns embeddings
# You DO need to understand WHAT embeddings are and WHY they work