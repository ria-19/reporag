"""
Topic: How Do LLMs Work? 

LLMs: The Final Piece of RAG
--------------------------------------------------
Key Concepts:
1. What is an LLM?
2. How does it generate text?
3. Why does RAG help?
4. Prompt engineering basics
--------------------------------------------------
"""

# --------------------------------------------------
# 1. LLM Basics (Simplified)
# --------------------------------------------------
"""
Large Language Model = Neural network trained on massive text 

Training:
- Input: Billions of words from internet
- Task: Predict next word
- Example: "The capital of France is ___" → "Paris"
- Random Guess -> Feedback via Training data: Loss calculation and backpropagation -> The Learning: gradient descent
- To get good at "predict the next word" when done on a cosmic scale (trillions of words), the model is forced to learn, from scratch, all the rules and patterns of human language and knowledge.

After training:
- Model learns grammar, facts, reasoning patterns
- Can generate human-like text
- But: Limited to training data, can hallucinate
"""

# --------------------------------------------------
# 2. Text Generation Process
# --------------------------------------------------
"""
How an LLM generates text (simplified):
 -  recursive, auto-regressive process, and work at token level
 
Given prompt: "Explain machine learning"

Step 1: Convert to tokens (subwords): tokenization
  "Explain machine learning" → [Explain, machine, learning]

Step 2: Model predicts probability of next token
 - The final step of the model (the "softmax" layer) outputs a probability score for every single token in its vocabulary.
  P("Machine") = 0.15
  P("Learning") = 0.08  
  P("is") = 0.12
  P("refers") = 0.05
  ...

Step 3: Sample next token (temperature controls randomness) from all tokens
  - Temperature 0: Always pick highest probability (deterministic)
  - Temperature 1: Sample based on probabilities (creative)

Step 4: Repeat until done: Input for next step - original prompt + new token
 - "Explain machine learning" → "Machine learning is..."
 
"""
# --------------------------------------------------
# Tokenization 
# --------------------------------------------------
"""
Tokenization = Converting raw text into integers the model can understand.

The Tokenizer:
- A separate but essential component trained alongside the LLM
- Acts as the model’s personal dictionary

Common Algorithm: Byte-Pair Encoding (BPE) or variants like WordPiece

How it works (Intuition):
1. Start with characters: A-Z, a-z, punctuation, etc.
2. Find most common adjacent character pair (e.g., 't' + 'h')
3. Merge them into a new token → "th"
4. Repeat with next frequent pairs: "er" → "th" + "e" → "the"
5. After thousands of merges → final vocabulary = mix of:
   - Whole words ("the", "cat")
   - Subwords ("ing", "ation")
   - Single characters

Why this works:
- Can represent any text, including unseen or rare words
- Keeps vocab compact (≈30K–100K tokens)

Stopping the Loop (During Generation):
- **EOS Token**: A special <|endoftext|> or </s> token marks the end of sequence.
  When predicted with highest probability → generation stops.
- **Max Length Limit**: A safety cutoff (max_tokens) stops generation even if EOS not reached.

Result:
- Efficient, flexible text representation
- Enables the model to “speak” in its own compressed token language
"""

# --------------------------------------------------
# 3. The Hallucination Problem
# --------------------------------------------------
"""
Without RAG:

User: "What is our company's PTO policy?"
LLM: [Makes up answer based on general knowledge]
     "Most companies offer 10-15 days..."
     ❌ WRONG! Not specific to YOUR company

With RAG:

User: "What is our company's PTO policy?"
System: 
  1. Search company docs
  2. Find: "Company offers 20 vacation days, 10 sick days..."
  3. Add to prompt as context
  4. LLM generates answer FROM CONTEXT
     "According to the company handbook, you get 20 vacation days..."
     ✅ CORRECT! Grounded in actual docs
"""

# --------------------------------------------------
# 4. Prompt Engineering for RAG
# --------------------------------------------------
"""
This is the final and most *artistic* part of building a RAG system.

Prompt Engineering = The bridge between:
- **Deterministic Retrieval** → factual, rule-based, reproducible
- **Probabilistic Generation** → creative, pattern-based, non-deterministic

It’s how we *constrain* the model’s creativity to get factual, reliable, and context-grounded answers.

--------------------------------------------------
# The "Why": The LLM Is a People-Pleaser, Not a Database
--------------------------------------------------
To understand prompt engineering, you must first adopt the *right mental model* of an LLM.

❌ **Bad Mental Model:**
> “The LLM is a knowledge engine. I give it data, it gives me truth.”

✅ **Good Mental Model:**
> “The LLM is a hyper-intelligent actor trained to play the role of a helpful assistant — 
> predicting the most probable next token based on the input script you give it.”

The model’s objective isn’t to be correct.
It’s to sound *plausibly correct*.

When your prompt is vague or missing clear constraints:
- The model’s “helpful assistant” instincts take over.
- It fills in missing info from its internal memory (parametric knowledge).
- Result → confident but fabricated answers (hallucinations).

--------------------------------------------------
# The "What": Prompt Engineering in Practice
--------------------------------------------------
Prompt engineering is the art + science of writing a *job description* for the LLM:
- What role it should play
- What data it can use (context)
- What to do when data is missing
- What format the output should take

You’re not just writing a question — you’re writing *instructions for behavior*. 

Bad Prompt:
  Context: [bunch of text]
  Question: What is X?

Problems:
- No instruction to use context
- No instruction to admit ignorance
- No structure

Good Prompt:
  You are a helpful assistant. Answer based ONLY on the context below.
  If the answer is not in the context, say "I don't have that information."
  
  Context:
  [Retrieved documents with sources]
  
  Question: What is X?
  
  Answer:

Why better:
- Clear instructions (use context only)
- Explicit failure mode (admit ignorance)
- Structured format
"""


# --------------------------------------------------
# Attention Mechanism & KV Cache
# --------------------------------------------------
"""
Attention = How the model decides *which parts of the input matter most* when predicting the next token.

Core Idea:
Each token doesn’t exist in isolation — it “looks back” at previous tokens and decides how much attention to give them.

For every token, the model computes three vectors:
- **Query (Q)**: “I am the current token. What should I pay attention to?”
- **Key (K)**: “I am a previous token. This is what I represent.”
- **Value (V)**: “I am a previous token. This is the information I hold.”

How It Works (Simplified):
1. For the current token, compare its **Q** against all previous **K** vectors → produces *attention weights*.
2. Use these weights to take a weighted sum of all **V** vectors → this becomes the contextual representation for the current token.
3. The model now “knows” which previous words are important for the next prediction.

Example:
“The cat sat on the mat.”
When predicting “mat,” the token “on” gets higher attention weight than “cat” or “the.”

--------------------------------------------------
# The Scaling Problem
--------------------------------------------------
Naive Approach:
- Every time a new token is added, recompute Q, K, and V for *all* tokens.
- So generating the 100th token requires reprocessing 99 previous ones.

This means:
- Computation grows *linearly per token* → O(n²) total for the full sequence.
- Example: 
  1st token → 1 comparison  
  2nd token → 2 comparisons  
  100th token → 100 comparisons  
  = 1 + 2 + 3 + ... + 100 ≈ 5,000 attention ops

Result: Expensive and slow for long sequences.

--------------------------------------------------
# KV Cache Optimization
--------------------------------------------------
Goal: Avoid recomputing Keys and Values for previous tokens.

Efficient Way (Using KV Cache):
1. When processing each token, compute its **K** and **V** once and store them in GPU memory (the KV cache).
2. On the next step:
   - Compute only the new token’s Q.
   - Reuse cached K/V for all prior tokens.
   - Append the new K/V to the cache for future use.

Effect:
- Computation becomes *linear* with sequence length → O(n).
- Enables real-time text generation and streaming responses.

--------------------------------------------------
# Why It Matters
--------------------------------------------------
- Attention gives LLMs the ability to understand relationships, context, and meaning across tokens.
- KV caching makes that power *scalable* and *efficient*.
- Together, they’re the reason modern transformers can generate coherent paragraphs, not just isolated words.
"""


# ---------------------------------------------------------
# Let's test different prompt styles
# ---------------------------------------------------------

def test_prompts(context, question):
    """Compare multiple prompt styles to illustrate RAG prompt quality."""
    
    # Prompt 1: Minimal (bad)
    prompt1 = f"{context}\n{question}"
    
    # Prompt 2: Basic instruction
    prompt2 = f"""Answer the question using the context below.
    
Context:
{context}

Question:
{question}

Answer:"""
    
    # Prompt 3: Detailed instruction (best)
    prompt3 = f"""You are a helpful assistant. Your task is to answer questions based ONLY on the provided context.

Rules:
1. Answer from context only
2. If unsure or information not in context, say so
3. Cite specific parts of context in your answer
4. Be concise but complete

Context:
{context}

Question:
{question}

Answer:"""
    
    return {"minimal": prompt1, "basic": prompt2, "detailed": prompt3}


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    context = "The company offers 20 vacation days per year."
    question = "How many vacation days do I get?"

    prompts = test_prompts(context, question)
    
    print("Prompt Comparison:")
    print("\n1. Minimal:")
    print(prompts["minimal"][:100] + "...")
    
    print("\n2. Basic:")
    print(prompts["basic"][:150] + "...")
    
    print("\n3. Detailed (BEST):")
    print(prompts["detailed"][:200] + "...")
    
"""
# Hands-On: Prompt Engineering Experiments (30 min)

EXERCISE: Test Different Prompts with Ollama

Goal:
- Observe how prompt phrasing affects answer quality and factual accuracy.
- Compare "no instruction", "basic instruction", and "structured instruction" prompts.

Pre-requisites:
- Ollama running locally (`ollama serve`)
- Model installed (e.g., `ollama pull llama2`)

Run:
$ python prompt_experiments.py
"""

import requests

# ---------------------------------------------------------
# LLM Call Helper
# ---------------------------------------------------------
def call_llm(prompt: str) -> str:
    """Call Ollama API locally and return the model's response."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 256}
        },
        timeout=60
    )

    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        return f"Error: {response.status_code} - {response.text}"


# ---------------------------------------------------------
# Test Scenario
# ---------------------------------------------------------
context = """
The company's remote work policy:
- Work from home up to 3 days per week
- Core hours: 10am–3pm local time
- All meetings must be recorded
"""

question = "Can I work fully remote?"


# ---------------------------------------------------------
# Test 1 — No Instruction (will likely hallucinate)
# ---------------------------------------------------------
prompt_bad = f"""Context:
{context}

Question:
{question}"""

answer_bad = call_llm(prompt_bad)
print("🔴 Test 1 - No instruction:")
print(answer_bad)
print("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------
# Test 2 — With Basic Instruction
# ---------------------------------------------------------
prompt_good = f"""Answer based ONLY on the context. 
If not in context, say so.

Context:
{context}

Question:
{question}

Answer:"""

answer_good = call_llm(prompt_good)
print("🟡 Test 2 - With instruction:")
print(answer_good)
print("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------
# Test 3 — With Structured Output Request (Best Practice)
# ---------------------------------------------------------
prompt_best = f"""You are a precise assistant. 
Answer based ONLY on the provided context.

Context:
{context}

Question:
{question}

Provide your answer in this format:
- Direct answer: [Yes/No/Partially]
- Explanation: [Why?]
- Source: [Quote from context]

Answer:"""

answer_best = call_llm(prompt_best)
print("🟢 Test 3 - Structured output:")
print(answer_best)
print("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------
# OBSERVATION NOTES
# ---------------------------------------------------------
"""
Observe:
1. Instruction drastically improves grounding.
2. Structured format produces clarity and consistency.
3. Lower temperatures (0.3–0.5) reduce creative drift.

Key Learnings:
- Always instruct model to use context ONLY.
- Always define a fallback behavior (say "I don't know").
- Request structured output when precision matters.
- Tune temperature for desired creativity vs. accuracy tradeoff.
"""
