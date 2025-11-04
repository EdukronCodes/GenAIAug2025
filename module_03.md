# Module 3: Representations & Search Algorithms

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 3-4

---

## Class 3: Embedding Models

### Topics Covered

- What are embeddings?
- Types: Word2Vec, GloVe, Sentence Transformers, OpenAI Embeddings
- Vector similarity (cosine, Euclidean)
- Use in semantic search and RAG systems

### Learning Objectives

By the end of this class, students will be able to:
- Understand the concept of embeddings and their importance
- Compare different embedding models and their characteristics
- Calculate vector similarity using cosine and Euclidean distance
- Apply embeddings in semantic search applications
- Integrate embeddings into RAG systems

### Core Concepts

#### What are Embeddings?

**Definition:**
- Numerical vector representations of text, images, or other data
- Capture semantic meaning in high-dimensional space
- Enable mathematical operations on meaning

**Key Properties:**
- Similar meanings → Similar vectors
- Preserve relationships (e.g., king - man + woman ≈ queen)
- Fixed-dimensional regardless of input length

**Code Example: Generating and Comparing Embeddings**

```python
import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Method 1: Using OpenAI Embeddings
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embedding(text, model="text-embedding-3-small"):
    """Generate embedding using OpenAI"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return np.array(response.data[0].embedding)

# Method 2: Using Sentence Transformers (local)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sentence_embedding(text):
    """Generate embedding using Sentence Transformers"""
    return model.encode(text)

# Compare embeddings
text1 = "Machine learning is a subset of AI"
text2 = "AI includes machine learning techniques"
text3 = "The weather is sunny today"

emb1 = get_sentence_embedding(text1)
emb2 = get_sentence_embedding(text2)
emb3 = get_sentence_embedding(text3)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_12 = cosine_similarity(emb1, emb2)  # Should be high
similarity_13 = cosine_similarity(emb1, emb3)  # Should be low

print(f"Similarity (ML texts): {similarity_12:.3f}")
print(f"Similarity (ML vs weather): {similarity_13:.3f}")
```

**Pro Tip:** Use OpenAI embeddings for production applications requiring high accuracy. Use Sentence Transformers for local/offline applications or when data privacy is critical.

**Common Pitfall:** Not normalizing embeddings before computing cosine similarity can lead to incorrect results. Always normalize vectors when using cosine similarity.

#### Types of Embeddings

**1. Word2Vec (2013)**
- Word-level embeddings
- Two architectures: Skip-gram and CBOW
- Context-based learning
- Limitations: Single vector per word, no context

**2. GloVe (Global Vectors, 2014)**
- Word-level embeddings
- Combines global and local statistics
- Matrix factorization approach
- Better performance on word analogy tasks

**3. Sentence Transformers (2019+)**
- Sentence/document-level embeddings
- Based on BERT, RoBERTa, or other transformers
- Fine-tuned for similarity tasks
- Examples: all-MiniLM-L6-v2, all-mpnet-base-v2

**4. OpenAI Embeddings**
- Text-embedding-ada-002 (current)
- Text-embedding-3-small, Text-embedding-3-large
- Optimized for semantic search
- 1536-dimensional vectors

**5. Multilingual Embeddings**
- Support for multiple languages
- Examples: multilingual-MiniLM, multilingual-mpnet

#### Vector Similarity

**Cosine Similarity:**
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```
- Range: -1 to 1 (typically 0 to 1 for embeddings)
- Measures angle between vectors
- Best for normalized embeddings
- Direction matters more than magnitude

**Euclidean Distance:**
```
distance = √(Σ(Aᵢ - Bᵢ)²)
```
- Range: 0 to ∞
- Measures straight-line distance
- Smaller distance = more similar
- Magnitude matters

**When to Use:**
- **Cosine Similarity:** Normalized embeddings, semantic similarity
- **Euclidean Distance:** Raw embeddings, when magnitude is important

**Code Example: Vector Similarity Calculations**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calculate_similarities(embeddings, query_embedding, metric='cosine'):
    """Calculate similarities between query and document embeddings"""
    if metric == 'cosine':
        # Cosine similarity (higher is better)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        return similarities
    elif metric == 'euclidean':
        # Euclidean distance (lower is better, convert to similarity)
        distances = euclidean_distances([query_embedding], embeddings)[0]
        # Convert to similarity (1 / (1 + distance))
        similarities = 1 / (1 + distances)
        return similarities

# Example usage
doc_embeddings = np.array([
    get_sentence_embedding("Python is a programming language"),
    get_sentence_embedding("Machine learning uses algorithms"),
    get_sentence_embedding("Python supports machine learning")
])

query_emb = get_sentence_embedding("programming with Python")

cosine_sim = calculate_similarities(doc_embeddings, query_emb, metric='cosine')
euclidean_sim = calculate_similarities(doc_embeddings, query_emb, metric='euclidean')

print("Cosine similarities:", cosine_sim)
print("Euclidean similarities:", euclidean_sim)
```

**Pro Tip:** Always use cosine similarity for semantic search with normalized embeddings. Euclidean distance is better when vector magnitude has meaning (e.g., TF-IDF vectors).

**Check Your Understanding:**
1. Why is cosine similarity preferred for semantic search?
2. What's the difference between word-level and sentence-level embeddings?
3. How do you choose between different embedding models?

#### Use in Semantic Search

**Semantic Search Process:**
1. Query embedding: Convert query to vector
2. Database search: Find similar vectors in database
3. Ranking: Sort by similarity score
4. Retrieval: Return top-k results

**Advantages over Keyword Search:**
- Understands synonyms
- Handles paraphrasing
- Captures semantic relationships
- Language-independent (with multilingual models)

#### Use in RAG Systems

**Embedding Role in RAG:**
- Document embeddings stored in vector database
- Query embedding used for retrieval
- Similarity threshold for filtering
- Hybrid approaches combine with keyword search

#### Training Embedding Models at Scale

Modern sentence embedding models are typically trained with contrastive objectives (e.g., InfoNCE) that pull positives together and push negatives apart. High-quality positives can be mined from parallel data, entailment pairs, or cross-encoder labels; hard negatives come from BM25/ANN nearest neighbors or in-batch negatives.

A practical pipeline uses a strong cross-encoder to relabel bi-encoder candidates (knowledge distillation), then fine-tunes the bi-encoder with batch-hard negatives. This balances retrieval efficiency and quality while keeping inference a single forward pass.

Training flow:
```
Corpus → Candidate Mining (BM25/ANN) → Cross‑Encoder Rerank (labels) →
Triplets/Pairs (q, pos, neg) → Bi‑Encoder Fine‑tuning (contrastive) → Export
```

#### Indexing, Filtering, and Maintenance

Choose ANN structures based on scale and update patterns: HNSW for high recall and dynamic inserts; IVF/IVFPQ for billion-scale with lower RAM; OPQ/PQ for compression. Use metadata filters (time, source, tenant) to scope retrieval and improve precision.

Plan for index refresh and drift: periodic re‑embedding after model updates, compaction to remove tombstones, and A/B evaluation (Recall@k, NDCG) to validate improvements. Size memory using ≈ N × (d × bytes_per_dim + overhead).

#### Evaluation and Benchmarks

Use BEIR/MTEB-style suites to assess generalization across domains (msmarco, scifact, fiqa, nq). Report Recall@k, MRR, and NDCG, plus latency at fixed recall targets. Track OOD performance to detect domain shift and plan domain adaptation or few-shot re‑tuning.

#### Multi‑Vector and Sparse‑Dense Hybrids

Late-interaction models (e.g., ColBERT) keep token-level vectors and compute MaxSim at query time for higher accuracy at additional memory/latency. SPLADE produces sparse expansions that combine well with BM25. Hybrid scoring (α·dense + (1−α)·sparse) often dominates either alone.

Embedding retrieval data path:
```
Text → Tokenize → Encoder → Vector (normalize) → ANN Index → top‑k IDs →
Metadata Filter → (Optional) Cross‑Encoder Rerank → Final Contexts
```

---

## Class 4: Overview of All Major LLMs

### Topics Covered

- GPT family (GPT-3.5, GPT-4, GPT-5)
- LLaMA, Falcon, Mistral, Claude, Gemini
- Key differences: architecture, context size, fine-tuning ability
- Choosing the right LLM for a use case

### Learning Objectives

By the end of this class, students will be able to:
- Identify major LLM families and their characteristics
- Compare LLMs based on architecture, size, and capabilities
- Understand context window limitations and implications
- Select appropriate LLMs for specific use cases
- Evaluate trade-offs between different models

### Core Concepts

#### GPT Family (OpenAI)

**GPT-3.5 (2022)**
- Variants: text-davinci-003, gpt-3.5-turbo
- 175B parameters (GPT-3 base)
- Context: 4K tokens (turbo), 16K tokens
- Strong general capabilities
- API-based access

**GPT-4 (2023)**
- Multimodal capabilities (text + images)
- 8K and 32K context windows
- Improved reasoning and instruction following
- GPT-4 Turbo: 128K context window
- More reliable and accurate

**GPT-4o (2024)**
- Optimized for speed and cost
- Multimodal (text, vision, audio)
- 128K context window
- Faster inference

#### LLaMA (Meta)

**LLaMA 1 (2023)**
- Open-source models: 7B, 13B, 33B, 65B parameters
- Self-hosted, requires significant compute
- Strong performance on benchmarks
- No API, requires hosting

**LLaMA 2 (2023)**
- Improved training and safety
- 7B, 13B, 70B variants
- Chat-optimized versions
- Commercial use allowed

**LLaMA 3 (2024)**
- 8B, 70B, 405B variants
- Improved reasoning
- Better instruction following
- Extended context windows

#### Claude (Anthropic)

**Claude 2 (2023)**
- 100K context window
- Strong safety and helpfulness
- Good for long documents
- API-based access

**Claude 3 (2024)**
- Variants: Haiku, Sonnet, Opus
- 200K context window
- Multimodal capabilities
- Improved reasoning

#### Gemini (Google)

**Gemini Pro (2023)**
- Multimodal from the ground up
- Strong reasoning capabilities
- Available via API
- Competitive performance

**Gemini 1.5 (2024)**
- 1M context window (experimental)
- Improved performance
- Better multimodal understanding

#### Other Notable Models

**Mistral AI**
- Mistral 7B, Mixtral 8x7B
- Open-source, efficient
- Strong performance per parameter

**Falcon (Technology Innovation Institute)**
- Falcon-40B, Falcon-180B
- Open-source
- Apache 2.0 license

### Key Comparison Criteria

#### Architecture
- **Decoder-only:** GPT, LLaMA, Falcon
- **Encoder-decoder:** T5, Flan-T5
- **Mixture of Experts:** Mixtral, GPT-4 (rumored)

#### Context Size
- **Small (4K-8K):** GPT-3.5-turbo, LLaMA 7B
- **Medium (32K-128K):** GPT-4, Claude 3 Sonnet
- **Large (200K+):** Claude 3, Gemini 1.5
- **Very Large (1M+):** Gemini 1.5 Pro (experimental)

#### Fine-tuning Ability
- **Full fine-tuning:** LLaMA, Mistral (self-hosted)
- **LoRA/QLoRA:** Most open-source models
- **API fine-tuning:** OpenAI (GPT-3.5), Anthropic (limited)
- **No fine-tuning:** Most API-based models (rely on prompt engineering)

#### Access Model
- **API-only:** GPT-4, Claude, Gemini
- **Self-hosted:** LLaMA, Mistral, Falcon
- **Both:** Some models available via API and for download

### Choosing the Right LLM

**Considerations:**

1. **Use Case Requirements**
   - Simple Q&A → GPT-3.5-turbo
   - Complex reasoning → GPT-4, Claude 3 Opus
   - Long documents → Claude 3, Gemini 1.5
   - Code generation → GPT-4, Claude 3

2. **Budget Constraints**
   - Cost-effective: GPT-3.5-turbo, LLaMA (self-hosted)
   - Premium: GPT-4, Claude 3 Opus

3. **Privacy Requirements**
   - High privacy: Self-hosted models (LLaMA, Mistral)
   - API acceptable: OpenAI, Anthropic, Google

4. **Fine-tuning Needs**
   - Custom domain: Self-hosted models with LoRA
   - General purpose: API models with prompt engineering

5. **Latency Requirements**
   - Fast: GPT-3.5-turbo, Claude 3 Haiku
   - Quality over speed: GPT-4, Claude 3 Opus

#### Pretraining, Tuning, and Decoding

Decoder‑only LMs (GPT/LLaMA) optimize next‑token prediction on large corpora; encoder‑decoder (T5) frame tasks text‑to‑text with strong supervised transfer. Instruction tuning aligns models to follow natural language commands; RLHF/constitutional AI further steers behavior toward helpful and harmless outputs.

Decoding controls quality and diversity: temperature, top‑k/p, and penalties manage repetition and creativity; constrained decoding (regex/JSON schemas) ensures structured outputs. KV‑cache, batching, and speculative decoding improve throughput and latency under load.

#### Tool Use, Function Calling, and RAG Synergy

Function calling grounds outputs in external tools (search, calculators, databases) and enforces schemas. RAG augments knowledge without retraining; routing policies can select models, tools, or prompts based on intent. For complex tasks, planner‑executor patterns decompose problems into tool‑invocation steps.

#### Efficiency: Quantization and PEFT

Quantization (INT8/INT4) reduces memory and increases throughput with modest quality costs; QLoRA combines 4‑bit base models with low‑rank adapters for efficient tuning. PEFT methods (LoRA, prefix/p‑tuning) adapt models with minimal parameters, enabling domain specialization on modest GPUs.

#### Evaluation, Safety, and Governance

Use diverse benchmarks (MMLU, MT‑Bench, HELM) plus domain‑specific tests. Track jailbreak resistance, prompt‑injection handling, and harmful output filters. Establish human‑in‑the‑loop review for high‑stakes tasks and maintain audit logs of prompts/outputs for compliance.

LLM serving flow:
```
Prompt (system+user+tools) → Safety/Policy Layer → LLM Decode (KV‑cache) →
(Optional) Function Calls/RAG → Post‑processing (parsers, validators) → Output
```

### Readings

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3
- "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
- Recent LLM architecture papers and comparisons

 

### Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [OpenAI Model Documentation](https://platform.openai.com/docs/models)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [LLM Comparison Tools](https://chat.lmsys.org/)

### Practical Code Examples

#### Complete Semantic Search Implementation

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]):
        """Add documents and generate embeddings"""
        self.documents = documents
        self.embeddings = self.model.encode(documents, show_progress_bar=True)
        print(f"Indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_embedding = self.model.encode(query)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (self.documents[i], float(similarities[i]))
            for i in top_indices
        ]
        return results

# Usage
search_engine = SemanticSearch()
documents = [
    "Python is a high-level programming language",
    "Machine learning algorithms learn from data",
    "Natural language processing analyzes text",
    "Deep learning uses neural networks"
]
search_engine.add_documents(documents)

results = search_engine.search("programming with Python", top_k=3)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc}")
```

#### LLM Selection Helper

```python
class LLMSelector:
    """Helper class for selecting appropriate LLM"""
    
    MODELS = {
        "gpt-3.5-turbo": {
            "context": 4096,
            "cost_per_1k_tokens": 0.0015,
            "speed": "fast",
            "best_for": ["simple Q&A", "chatbots", "content generation"]
        },
        "gpt-4": {
            "context": 8192,
            "cost_per_1k_tokens": 0.03,
            "speed": "medium",
            "best_for": ["complex reasoning", "code generation", "analysis"]
        },
        "claude-3-opus": {
            "context": 200000,
            "cost_per_1k_tokens": 0.015,
            "speed": "slow",
            "best_for": ["long documents", "analysis", "research"]
        }
    }
    
    @classmethod
    def recommend(cls, use_case: str, budget: str = "medium", 
                   context_size: int = 4096, privacy: bool = False):
        """Recommend LLM based on requirements"""
        recommendations = []
        
        for model, specs in cls.MODELS.items():
            score = 0
            
            # Context size check
            if specs["context"] >= context_size:
                score += 2
            else:
                continue  # Skip if context too small
            
            # Budget check
            if budget == "low" and specs["cost_per_1k_tokens"] < 0.01:
                score += 2
            elif budget == "medium":
                score += 1
            
            # Use case match
            if use_case.lower() in " ".join(specs["best_for"]).lower():
                score += 2
            
            recommendations.append((model, score, specs))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[0] if recommendations else None

# Usage
recommendation = LLMSelector.recommend(
    use_case="complex reasoning",
    budget="medium",
    context_size=8000
)
print(f"Recommended: {recommendation[0]}")
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low similarity scores** | All embeddings have low similarity | Check normalization, try different embedding model, verify text preprocessing |
| **Embedding dimension mismatch** | Cannot compare embeddings | Ensure all embeddings use same model and dimension |
| **Slow embedding generation** | High latency | Use faster models (all-MiniLM vs mpnet), batch processing, caching |
| **Poor search results** | Irrelevant documents retrieved | Try different embedding models, add reranking, adjust similarity threshold |
| **Context window overflow** | Token limit exceeded | Chunk documents, use summarization, select model with larger context |
| **API rate limits** | 429 errors from embedding API | Implement caching, use local models, add retry logic |

**Common Pitfalls:**
- **Pitfall:** Using word-level embeddings for sentence/document search
  - **Solution:** Always use sentence/document-level embeddings for semantic search
- **Pitfall:** Not normalizing embeddings before similarity calculation
  - **Solution:** Normalize all embeddings to unit vectors when using cosine similarity
- **Pitfall:** Choosing wrong LLM for context size requirements
  - **Solution:** Always check context window size before selecting model

### Quick Reference Guide

#### Embedding Model Comparison

| Model | Dimensions | Type | Best For | Speed |
|-------|------------|------|----------|-------|
| text-embedding-3-small | 1536 | API | Production apps | Fast |
| text-embedding-3-large | 3072 | API | High accuracy | Medium |
| all-MiniLM-L6-v2 | 384 | Local | Fast local search | Very Fast |
| all-mpnet-base-v2 | 768 | Local | High accuracy | Medium |
| multilingual-MiniLM | 384 | Local | Multilingual | Fast |

#### LLM Selection Matrix

| Requirement | Recommended Model | Alternative |
|-------------|------------------|------------|
| Simple Q&A | GPT-3.5-turbo | Claude Haiku |
| Complex reasoning | GPT-4 | Claude Opus |
| Long documents | Claude 3 | Gemini 1.5 |
| Code generation | GPT-4 | Claude 3 |
| Cost-sensitive | GPT-3.5-turbo | LLaMA (self-hosted) |
| Privacy-critical | LLaMA (self-hosted) | Mistral (self-hosted) |

### Case Studies

#### Case Study: Embedding Model Migration

**Challenge:** A company needed to improve search accuracy in their knowledge base.

**Initial Setup:**
- Used TF-IDF for keyword search
- 40% irrelevant results
- No semantic understanding

**Solution:**
- Migrated to OpenAI text-embedding-3-small
- Implemented hybrid search (BM25 + embeddings)
- Added reranking with cross-encoder

**Results:**
- 85% relevant results (40% → 85%)
- 50% reduction in search time
- Improved user satisfaction

**Lessons Learned:**
- Hybrid search outperforms pure semantic search
- Reranking significantly improves precision
- Cost of embeddings offset by improved accuracy

### Hands-On Lab: Build a Semantic Search System

**Lab Objective:** Create a complete semantic search system with embeddings and similarity search.

**Steps:**

1. **Setup**
```bash
pip install sentence-transformers numpy scikit-learn
```

2. **Implement Search Engine**
```python
# Use code examples above
# Add document indexing
# Implement search interface
```

3. **Test and Evaluate**
```python
# Test with sample queries
# Measure retrieval accuracy
# Compare different embedding models
```

**Expected Outcomes:**
- Working semantic search system
- Understanding of embedding models
- Knowledge of similarity metrics
- Ability to optimize search performance

### Glossary

**Embedding:** A dense vector representation of text, images, or other data that captures semantic meaning in high-dimensional space.

**Cosine Similarity:** A metric measuring the cosine of the angle between two vectors, used for comparing normalized embeddings.

**Euclidean Distance:** The straight-line distance between two vectors in Euclidean space, smaller distance indicates more similarity.

**Sentence Transformer:** A model architecture fine-tuned to produce sentence-level embeddings optimized for similarity tasks.

**Context Window:** The maximum number of tokens a language model can process in a single input/output sequence.

**Fine-tuning:** Adapting a pre-trained model to a specific task using task-specific training data.

**LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that adds trainable low-rank matrices to model weights.

**Reranking:** A second-stage retrieval process that re-orders candidates using a more expensive but accurate model.

**Semantic Search:** Search method that understands meaning and intent rather than just matching keywords.

**Vector Database:** A specialized database optimized for storing and querying high-dimensional vector embeddings.

### Key Takeaways

1. Embeddings enable semantic understanding and search beyond keywords
2. Different embedding models serve different purposes (word vs. sentence level)
3. Vector similarity metrics (cosine, Euclidean) are fundamental to retrieval
4. LLM selection depends on use case, budget, privacy, and requirements
5. Context window size is a critical factor for long-document applications
6. Open-source models offer flexibility but require infrastructure
7. Hybrid search (keyword + semantic) often outperforms either alone
8. Proper normalization is crucial for accurate similarity calculations
9. Reranking significantly improves retrieval precision
10. Cost, latency, and accuracy must be balanced in production systems

---

**Previous Module:** [Module 2: GenAI Project Architecture & Flow](../module_02.md)  
**Next Module:** [Module 4: Search Algorithms & Retrieval Techniques](../module_04.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

