# Module 6: RAG & Transformer Architecture

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 10-11

---

## Class 10: RAG (Retrieval-Augmented Generation)

### Topics Covered

- Architecture & Workflow
- Retriever + Generator pipeline
- Evaluation of RAG systems
- Detailed notes on hybrid retrieval (BM25 + Embedding)

### Learning Objectives

By the end of this class, students will be able to:
- Understand complete RAG architecture and workflow
- Design retriever and generator components
- Evaluate RAG system performance
- Implement hybrid retrieval strategies
- Optimize RAG systems for production

### Core Concepts

#### RAG Architecture & Workflow

**Complete RAG Pipeline:**

```
1. Data Ingestion
   ├─ Document Loading
   ├─ Text Preprocessing
   ├─ Chunking
   └─ Embedding Generation

2. Indexing
   ├─ Vector Store Creation
   ├─ Metadata Storage
   └─ Index Optimization

3. Query Processing
   ├─ Query Embedding
   ├─ Retrieval
   ├─ Reranking (optional)
   └─ Context Assembly

4. Generation
   ├─ Prompt Construction
   ├─ LLM Inference
   └─ Response Post-processing
```

#### Retriever + Generator Pipeline

**Retriever Component:**
- **Input:** User query
- **Process:** 
  - Query embedding
  - Similarity search
  - Top-k retrieval
  - Optional reranking
- **Output:** Relevant document chunks

**Generator Component:**
- **Input:** Query + Retrieved context
- **Process:**
  - Prompt template construction
  - LLM inference
  - Response generation
- **Output:** Final answer

**Integration:**
- Context window management
- Token budgeting
- Prompt optimization
- Quality control

#### Advanced RAG Techniques

**1. Query Expansion**
- Generate query variations
- Improve retrieval recall
- Handle synonyms and paraphrasing

**2. Reranking**
- Use cross-encoder models
- Improve retrieval precision
- Reorder top-k results

**3. Multi-Query Retrieval**
- Generate multiple queries
- Retrieve from each
- Combine results

**4. Parent Document Retrieval**
- Retrieve small chunks
- Return parent documents
- Better context preservation

**5. Query Routing**
- Route to different indexes
- Specialized retrieval strategies
- Domain-specific indexes

#### Hybrid Retrieval (BM25 + Embedding)

**Why Hybrid?**
- BM25: Excellent for exact keywords
- Embeddings: Excellent for semantic meaning
- Combined: Best of both worlds

**Implementation:**
```python
# 1. BM25 retrieval
bm25_results = bm25_retriever.retrieve(query, k=10)

# 2. Embedding retrieval
embedding_results = vector_store.similarity_search(query, k=10)

# 3. Combine and rerank
combined_results = combine_results(bm25_results, embedding_results)
reranked = reranker.rerank(query, combined_results)

# 4. Select top-k
final_context = reranked[:k]
```

**Scoring Strategies:**
- **Reciprocal Rank Fusion (RRF):**
  ```
  score = Σ(1 / (k + rank))
  ```
- **Weighted Combination:**
  ```
  score = α × bm25_score + (1-α) × embedding_score
  ```
- **Learned Combination:**
  - Train a model to combine scores
  - Optimal weights learned from data

**Benefits:**
- Higher recall (more relevant docs found)
- Better precision (more relevant docs ranked higher)
- Handles both exact and semantic matches
- More robust to query variations

#### Evaluation of RAG Systems

**Evaluation Metrics:**

**1. Retrieval Metrics:**
- **Precision@k:** Fraction of retrieved docs that are relevant
- **Recall@k:** Fraction of relevant docs that are retrieved
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant doc
- **NDCG (Normalized Discounted Cumulative Gain):** Ranking quality

**2. Generation Metrics:**
- **BLEU:** N-gram overlap with reference
- **ROUGE:** Recall-oriented summary evaluation
- **METEOR:** Semantic similarity
- **BERTScore:** Semantic similarity using embeddings

**3. End-to-End Metrics:**
- **Answer Accuracy:** Correctness of final answer
- **Faithfulness:** Answer grounded in retrieved context
- **Answer Relevance:** Answer relevance to question
- **Context Utilization:** How well context is used

**Evaluation Framework:**
```python
# 1. Create evaluation dataset
questions = ["What is X?", "How does Y work?"]
ground_truth = ["Expected answers"]
contexts = ["Relevant documents"]

# 2. Evaluate retrieval
retrieval_scores = evaluate_retrieval(
    questions, contexts, ground_truth
)

# 3. Evaluate generation
generation_scores = evaluate_generation(
    questions, contexts, ground_truth, answers
)

# 4. Evaluate end-to-end
rag_scores = evaluate_rag(
    questions, answers, ground_truth
)
```

**Common Issues:**
- **Hallucination:** Model generates information not in context
- **Insufficient Context:** Not enough relevant information retrieved
- **Context Overload:** Too much irrelevant context
- **Query Mismatch:** Query doesn't match document style

---

## Class 11: Transformer Architecture Deep Dive

### Topics Covered

- Attention mechanism (Self, Cross, Multi-head)
- Positional encoding, residual connections
- Encoder-Decoder models: BERT, GPT, T5
- Visual walkthrough & code snippets

### Learning Objectives

By the end of this class, students will be able to:
- Understand transformer architecture in detail
- Explain attention mechanisms mathematically
- Compare different transformer architectures
- Implement transformer components from scratch
- Understand how transformers enable modern LLMs

### Core Concepts

#### Transformer Architecture Overview

**Key Innovation:**
- Attention mechanism replaces recurrence
- Parallel processing of sequences
- Enables training on large datasets
- Foundation of modern LLMs

**Basic Structure:**
```
Input → Embedding → Positional Encoding
         ↓
    [N × Transformer Block]
         ↓
    Output Layer
```

#### Attention Mechanism

**Self-Attention:**
- Computes relationships within a sequence
- Each position attends to all positions
- Enables long-range dependencies

**Mathematical Formulation:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- d_k = dimension of keys
```

**Intuition:**
- **Query:** "What am I looking for?"
- **Key:** "What do I contain?"
- **Value:** "What information do I provide?"
- **Attention:** How much to weight each value

**Cross-Attention:**
- Attention between two sequences
- Used in encoder-decoder architectures
- Decoder attends to encoder outputs

**Multi-Head Attention:**
- Multiple attention mechanisms in parallel
- Captures different types of relationships
- Concatenated and projected

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

**Benefits:**
- Different representation subspaces
- More expressive
- Better parallelization

#### Positional Encoding

**Problem:**
- Attention is permutation-invariant
- Need to encode sequence order

**Solutions:**

**1. Sinusoidal Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Fixed, not learned
- Can extrapolate to longer sequences
- Used in original Transformer

**2. Learned Positional Embeddings:**
- Learned parameters
- Better for training data lengths
- Used in BERT, GPT

**3. Relative Position Encoding:**
- Encodes relative positions
- Better for longer sequences
- Used in some variants

#### Residual Connections

**Purpose:**
- Enable gradient flow in deep networks
- Facilitate training
- Allow identity mapping

**Structure:**
```
output = LayerNorm(x + Sublayer(x))
```

**Benefits:**
- Easier optimization
- Deeper networks possible
- Better performance

#### Encoder-Decoder Models

**BERT (Bidirectional Encoder Representations):**
- Encoder-only architecture
- Bidirectional attention
- Masked language modeling
- Good for classification, understanding

**GPT (Generative Pre-trained Transformer):**
- Decoder-only architecture
- Causal (masked) attention
- Autoregressive generation
- Good for generation tasks

**T5 (Text-to-Text Transfer Transformer):**
- Encoder-decoder architecture
- All tasks as text-to-text
- Unified framework
- Good for many NLP tasks

**Comparison:**

| Model | Architecture | Attention | Best For |
|-------|-------------|-----------|----------|
| BERT | Encoder | Bidirectional | Understanding |
| GPT | Decoder | Causal | Generation |
| T5 | Encoder-Decoder | Both | Translation, Summarization |

#### Implementation Overview

**Key Components:**

**1. Embedding Layer:**
```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
    
    def forward(self, x):
        return self.embedding(x) + self.pos_encoding(x)
```

**2. Multi-Head Attention:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Split into heads, compute attention, concatenate
        ...
```

**3. Transformer Block:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
```

### Readings

- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2

 

### Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)

### Practical Code Examples

#### Complete RAG Evaluation System

```python
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGEvaluator:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_retrieval(self, query: str, retrieved_docs: List[str], 
                          relevant_docs: List[str], k: int = 5):
        """Evaluate retrieval quality"""
        # Calculate recall@k
        recall = len(set(retrieved_docs[:k]) & set(relevant_docs)) / len(relevant_docs)
        
        # Calculate precision@k
        precision = len(set(retrieved_docs[:k]) & set(relevant_docs)) / k
        
        return {"recall@k": recall, "precision@k": precision}
    
    def evaluate_answer_quality(self, answer: str, reference: str):
        """Evaluate answer quality using embeddings"""
        answer_emb = self.embedding_model.encode(answer)
        ref_emb = self.embedding_model.encode(reference)
        
        similarity = cosine_similarity([answer_emb], [ref_emb])[0][0]
        
        return {"semantic_similarity": float(similarity)}
    
    def evaluate_faithfulness(self, answer: str, context: str):
        """Check if answer is grounded in context"""
        answer_emb = self.embedding_model.encode(answer)
        context_emb = self.embedding_model.encode(context)
        
        similarity = cosine_similarity([answer_emb], [context_emb])[0][0]
        
        return {"faithfulness_score": float(similarity)}

# Usage
evaluator = RAGEvaluator()
retrieval_metrics = evaluator.evaluate_retrieval(
    query="What is machine learning?",
    retrieved_docs=["doc1", "doc2", "doc3"],
    relevant_docs=["doc1", "doc2"],
    k=3
)
print(f"Retrieval metrics: {retrieval_metrics}")
```

**Pro Tip:** Always evaluate retrieval and generation separately. Good retrieval is necessary but not sufficient for good RAG performance.

**Common Pitfall:** Evaluating only end-to-end metrics can hide retrieval issues. Always measure retrieval recall independently.

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low retrieval recall** | Missing relevant documents | Increase k, improve embeddings, use hybrid search |
| **Hallucinated answers** | Answers not in context | Add faithfulness checks, improve prompts, use reranking |
| **Slow RAG pipeline** | High latency | Cache embeddings, optimize retrieval, use faster models |
| **Context overflow** | Token limit exceeded | Reduce chunk size, implement summarization, use token budgeting |
| **Poor answer quality** | Irrelevant or wrong answers | Improve retrieval, add reranking, optimize prompts |

### Quick Reference Guide

#### Transformer Architecture Comparison

| Architecture | Type | Use Case | Example Models |
|--------------|------|----------|----------------|
| Encoder-only | Bidirectional | Understanding, classification | BERT, RoBERTa |
| Decoder-only | Autoregressive | Generation, completion | GPT, LLaMA |
| Encoder-Decoder | Sequence-to-sequence | Translation, summarization | T5, BART |

#### RAG Evaluation Metrics

| Metric | Purpose | Range | When to Use |
|--------|---------|-------|-------------|
| Recall@k | Retrieval quality | 0-1 | Measure retrieval coverage |
| Precision@k | Retrieval accuracy | 0-1 | Measure retrieval relevance |
| Faithfulness | Answer grounding | 0-1 | Check hallucination |
| Semantic Similarity | Answer quality | 0-1 | Compare answers |

### Case Studies

#### Case Study: RAG System for Legal Document Analysis

**Challenge:** Law firm needed to search and analyze 50,000+ legal documents.

**Solution:** Implemented RAG with:
- Hybrid retrieval (BM25 + embeddings)
- GPT-4 for generation
- Custom evaluation framework

**Results:**
- 90% retrieval recall
- 85% answer accuracy
- 10x faster than manual review

**Lessons Learned:**
- Domain-specific embeddings crucial
- Evaluation framework essential
- Hybrid retrieval outperformed single method

### Key Takeaways

1. RAG combines retrieval and generation for knowledge-intensive tasks
2. Hybrid retrieval (BM25 + embeddings) provides best results
3. Transformer architecture enables modern LLMs through attention
4. Attention mechanism allows parallel processing and long-range dependencies
5. Different architectures (BERT, GPT, T5) serve different purposes
6. Proper evaluation is crucial for RAG system optimization
7. Understanding transformer internals helps in model selection and optimization
8. Separate evaluation of retrieval and generation provides better insights
9. Faithfulness checks prevent hallucination
10. Continuous monitoring and evaluation improve system performance

---

**Previous Module:** [Module 5: Frameworks for Building GenAI Applications](../module_05.md)  
**Next Module:** [Module 7: Tokenization & Embeddings in LLMs](../module_07.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

