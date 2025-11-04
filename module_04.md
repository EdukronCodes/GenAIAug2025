# Module 4: Search Algorithms & Retrieval Techniques

**Course:** Generative AI & Agentic AI  
**Module Duration:** 2 weeks  
**Classes:** 5-7

---

## Class 5: Search Algorithms — Fundamentals

### Topics Covered

- Keyword search vs Semantic search
- TF-IDF & BM25 introduction
- Dense vs Sparse retrieval
- When to use each approach

### Learning Objectives

By the end of this class, students will be able to:
- Distinguish between keyword and semantic search
- Understand TF-IDF and its applications
- Compare dense and sparse retrieval methods
- Choose appropriate search strategy for different scenarios

### Core Concepts

#### Keyword Search vs Semantic Search

**Keyword Search (Lexical Search):**
- Matches exact words or phrases
- Fast and efficient
- Requires exact term matching
- Examples: SQL LIKE, grep, traditional databases

**Semantic Search:**
- Understands meaning and intent
- Handles synonyms and paraphrasing
- Uses embeddings and vector similarity
- Examples: Embedding-based search, neural search

**Comparison:**

| Aspect | Keyword Search | Semantic Search |
|--------|---------------|-----------------|
| Speed | Very Fast | Moderate |
| Exact Match | Required | Not Required |
| Synonyms | No | Yes |
| Paraphrasing | No | Yes |
| Implementation | Simple | Complex |
| Cost | Low | Higher |

#### Query Understanding and Normalization

Robust retrieval starts with clean, well‑formed queries. Normalize case and Unicode, remove boilerplate, correct spelling, and expand common abbreviations. Apply language detection to select the appropriate analyzer/tokenizer (e.g., morphological analyzers for agglutinative languages). Use domain ontologies to map user jargon to canonical terms, and pre‑apply filters (time range, document type) to reduce the search space before scoring.

Pipeline sketch:
```
User Input → Normalize → Tokenize/Stem → Synonym/Thesaurus Expand → Filter → Candidate Generation → Scoring
```

#### Query Understanding and Normalization

Effective retrieval begins before scoring. Normalize inputs (Unicode NFC, case folding), strip boilerplate, and detect language and script to select the right tokenizer/embedding model. Apply query rewriting when appropriate: synonym expansion, typo correction, lemmatization, and phrase detection. For enterprise content, exploit metadata constraints (tenant, region, document type) to prune the search space early.

Query interpretation flow:
```
Raw Query → Normalize/Tokenize → Detect Language → Expand (synonyms, acronyms) →
Apply Filters (time, tags, access) → Candidate Generation → Scoring
```

#### TF-IDF (Term Frequency-Inverse Document Frequency)

**Definition:**
- Statistical measure to evaluate word importance
- Combines term frequency (TF) and inverse document frequency (IDF)

**Formula:**
```
TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
IDF(t, D) = log(Total documents / Documents containing term t)
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

**Characteristics:**
- Higher weight for frequent terms in document
- Lower weight for common terms across corpus
- Good for keyword-based retrieval
- Fast computation

**Use Cases:**
- Document search
- Information retrieval
- Text classification
- Feature extraction

#### Dense vs Sparse Retrieval

**Sparse Retrieval:**
- Uses sparse vectors (mostly zeros)
- Each dimension represents a word/token
- Examples: TF-IDF, BM25, keyword matching
- Fast and memory-efficient
- Good for exact keyword matching

**Dense Retrieval:**
- Uses dense vectors (embeddings)
- Continuous-valued vectors
- Examples: Sentence transformers, neural embeddings
- Captures semantic meaning
- Requires more computation

**Hybrid Approach:**
- Combines both methods
- Best of both worlds
- Higher accuracy
- More complex implementation

**Code Example: TF-IDF Implementation**

```python
import math
from collections import Counter
from typing import List, Dict

class TFIDF:
    def __init__(self, documents: List[str]):
        self.documents = [doc.lower().split() for doc in documents]
        self.vocabulary = self._build_vocabulary()
        self.idf_scores = self._calculate_idf()
    
    def _build_vocabulary(self):
        """Build vocabulary from all documents"""
        vocab = set()
        for doc in self.documents:
            vocab.update(doc)
        return sorted(vocab)
    
    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate IDF scores for all terms"""
        n_docs = len(self.documents)
        idf = {}
        
        for term in self.vocabulary:
            docs_with_term = sum(1 for doc in self.documents if term in doc)
            idf[term] = math.log(n_docs / (1 + docs_with_term))
        
        return idf
    
    def tf(self, term: str, document: List[str]) -> float:
        """Calculate term frequency"""
        return document.count(term) / len(document)
    
    def tfidf(self, term: str, document: List[str]) -> float:
        """Calculate TF-IDF score"""
        tf_score = self.tf(term, document)
        idf_score = self.idf_scores.get(term, 0)
        return tf_score * idf_score
    
    def vectorize(self, document: List[str]) -> List[float]:
        """Convert document to TF-IDF vector"""
        return [self.tfidf(term, document) for term in self.vocabulary]

# Usage
docs = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks",
    "machine learning algorithms learn from data"
]

tfidf = TFIDF(docs)
query = "machine learning".split()
query_vector = tfidf.vectorize(query)
print(f"Query vector: {query_vector[:5]}")  # First 5 dimensions
```

**Pro Tip:** Use scikit-learn's TfidfVectorizer for production applications. It's optimized and handles edge cases better than custom implementations.

**Common Pitfall:** Not handling empty documents or missing terms can cause division by zero errors. Always add smoothing (e.g., +1 in denominator) or handle edge cases.

---

## Class 6: BM25 Algorithm Deep Dive

### Topics Covered

- Formula and parameters (k1, b)
- Implementation in Python
- Comparing BM25 vs Embedding search
- Optimization techniques

### Learning Objectives

By the end of this class, students will be able to:
- Understand BM25 algorithm mathematically
- Implement BM25 from scratch
- Tune BM25 parameters for optimal performance
- Compare BM25 with embedding-based search

### Core Concepts

#### BM25 (Best Matching 25)

**Definition:**
- Probabilistic ranking function
- Improved version of TF-IDF
- Handles term frequency saturation
- Industry standard for keyword search

**Formula:**
```
BM25(q, d) = Σ IDF(qᵢ) × (f(qᵢ, d) × (k1 + 1)) / (f(qᵢ, d) + k1 × (1 - b + b × (|d| / avgdl)))

Where:
- q = query
- d = document
- f(qᵢ, d) = frequency of query term i in document d
- |d| = document length
- avgdl = average document length
- k1 = term frequency saturation parameter (default: 1.2)
- b = length normalization parameter (default: 0.75)
```

#### Parameters Explained

**k1 (Term Frequency Saturation):**
- Controls how quickly term frequency saturates
- Lower k1 → faster saturation (less emphasis on repeated terms)
- Higher k1 → slower saturation (more emphasis on repeated terms)
- Typical range: 0.5 - 2.0
- Default: 1.2

**b (Length Normalization):**
- Controls document length normalization
- b = 0 → no normalization
- b = 1 → full normalization
- Typical range: 0.0 - 1.0
- Default: 0.75

**Impact of Parameters:**
- **k1 = 0.5, b = 0.75:** Less emphasis on term frequency
- **k1 = 1.2, b = 0.75:** Standard settings
- **k1 = 2.0, b = 0.75:** More emphasis on term frequency
- **k1 = 1.2, b = 0.0:** No length normalization

##### Derivation & Intuition

BM25 emerges from the probabilistic relevance framework assuming term independence and a saturation function on term frequency. It balances how often a term appears in a document (TF) with how discriminative it is across the corpus (IDF), while correcting for document length via the b parameter to avoid bias toward long documents.

##### Practical Tuning Tips

- Start with k1≈0.9–1.2 and b≈0.6–0.9; grid‑search on a validation set with Recall@k/NDCG.
- Build per‑field BM25 with field boosts (e.g., title×2, headings×1.2, body×1.0) when using fielded indexes.
- Use stopword lists and minimal stemming; over‑aggressive stemming can harm precision.
- For multilingual corpora, use language‑specific analyzers and per‑language IDF statistics.

#### Implementation Considerations

**Preprocessing:**
- Tokenization
- Lowercasing (optional)
- Stop word removal (optional)
- Stemming/Lemmatization (optional)

**Efficiency:**
- Pre-compute IDF values
- Use inverted index for fast lookup
- Cache document lengths
- Vectorized operations in Python

#### BM25 vs Embedding Search

**Performance Comparison:**

| Aspect | BM25 | Embedding Search |
|--------|------|------------------|
| Exact Keywords | Excellent | Good |
| Synonyms | Poor | Excellent |
| Paraphrasing | Poor | Excellent |
| Speed | Very Fast | Moderate |
| Memory | Low | High |
| Setup Complexity | Low | High |

**When to Use BM25:**
- Exact keyword matching important
- Fast retrieval required
- Limited computational resources
- Domain-specific terminology

**When to Use Embeddings:**
- Semantic understanding needed
- Synonym and paraphrasing handling
- Cross-lingual search
- Multimodal search

**Hybrid Approach:**
- Combine BM25 and embedding scores
- Weighted combination
- Typically: 70% semantic + 30% keyword
- Best overall performance

**Code Example: BM25 Implementation**

```python
import math
from collections import Counter
from typing import List, Dict

class BM25:
    def __init__(self, documents: List[str], k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.idf_scores = self._calculate_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def _calculate_idf(self) -> Dict[str, float]:
        """Calculate IDF for all terms"""
        n_docs = len(self.documents)
        idf = {}
        
        for doc in self.documents:
            for term in set(doc):
                idf[term] = idf.get(term, 0) + 1
        
        # Calculate IDF
        for term in idf:
            idf[term] = math.log((n_docs - idf[term] + 0.5) / (idf[term] + 0.5) + 1.0)
        
        return idf
    
    def _calculate_score(self, query: List[str], doc_index: int) -> float:
        """Calculate BM25 score for query-document pair"""
        score = 0.0
        doc = self.documents[doc_index]
        doc_length = self.doc_lengths[doc_index]
        
        term_freq = Counter(doc)
        
        for term in query:
            if term not in self.idf_scores:
                continue
            
            tf = term_freq.get(term, 0)
            idf = self.idf_scores[term]
            
            # BM25 formula
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += numerator / denominator
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search documents and return top-k results"""
        query_tokens = self._tokenize(query)
        scores = []
        
        for i in range(len(self.documents)):
            score = self._calculate_score(query_tokens, i)
            scores.append((i, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Usage
docs = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for pattern recognition",
    "machine learning algorithms learn from data to make predictions"
]

bm25 = BM25(docs, k1=1.2, b=0.75)
results = bm25.search("machine learning", top_k=2)

for doc_idx, score in results:
    print(f"Score: {score:.3f} - {docs[doc_idx]}")
```

**Pro Tip:** Tune k1 and b parameters based on your corpus. For long documents, increase b. For term frequency emphasis, increase k1. Use grid search with validation set.

**Common Pitfall:** Using default parameters without tuning can lead to suboptimal results. Always validate parameter choices on your specific dataset.

#### Multi‑Stage Retrieval and Reranking

In large collections, first use a high‑recall generator (BM25 or ANN) to pull top‑k candidates, then apply a more expensive reranker (e.g., cross‑encoder, LambdaMART). This narrows candidates to a high‑quality top‑N for the LLM, improving both latency and answer quality.

Two‑stage flow:
```
Query → [BM25/ANN] (k=500) → candidate set → Cross‑Encoder rerank (top=50) → LLM context
```

Track the *recall at k* of the first stage and the *precision@N* after rerank. Adjust k and reranker depth to balance cost and quality.

---

## Class 7: HNSW (Hierarchical Navigable Small World)

### Topics Covered

- Introduction to approximate nearest neighbor search
- Graph-based retrieval
- HNSW algorithm principles
- Use cases in FAISS, Milvus, Weaviate, ChromaDB

### Learning Objectives

By the end of this class, students will be able to:
- Understand approximate nearest neighbor search
- Explain HNSW algorithm and its advantages
- Compare different vector databases
- Choose appropriate vector database for use case

### Core Concepts

#### Approximate Nearest Neighbor (ANN) Search

**Problem:**
- Exact nearest neighbor search is O(n) for n vectors
- Too slow for large-scale applications
- Need approximate solutions with acceptable accuracy

**Approaches:**
- **Tree-based:** KD-tree, Ball tree
- **Hash-based:** LSH (Locality Sensitive Hashing)
- **Graph-based:** HNSW, NSG
- **Quantization:** Product quantization, Scalar quantization

#### HNSW (Hierarchical Navigable Small World)

**Definition:**
- Multi-layer graph structure
- Fast approximate nearest neighbor search
- Logarithmic search complexity
- High accuracy

**Key Concepts:**

**1. Small World Property:**
- Most nodes can be reached in few hops
- Like social networks (six degrees of separation)
- Enables efficient navigation

**2. Hierarchical Structure:**
- Multiple layers with different densities
- Bottom layer: all nodes
- Higher layers: fewer nodes, longer edges
- Search starts at top layer, navigates down

**3. Search Process:**
```
1. Start at top layer (sparsest)
2. Find nearest neighbor
3. Move to next layer
4. Refine search
5. Continue to bottom layer
6. Return k nearest neighbors
```

**Advantages:**
- Fast: O(log n) search time
- Accurate: High recall
- Scalable: Handles millions of vectors
- Memory efficient: Graph structure

**Parameters:**
- **M:** Number of connections per node (default: 16)
- **ef_construction:** Controls graph quality (default: 200)
- **ef_search:** Controls search quality (default: 50)

#### Build, Updates, and Scaling

HNSW supports dynamic inserts; however, deletions typically mark tombstones and require periodic rebuild/compaction to reclaim memory and maintain search quality. For billion‑scale, use sharded HNSW or IVFPQ‑HNSW hybrids; balance M and ef_* to meet recall targets within latency budgets. Monitor graph connectivity (avg degree) and layer sizes.

#### Recall/Latency Trade‑offs

Increasing ef_search improves recall but grows query time roughly O(ef). Use per‑query adaptive ef based on early stopping criteria (e.g., when best distance stabilizes). For tail‑latency control, cap ef and rely on reranking to recover precision.

Search loop (simplified):
```
enter top layer at random node
for level L..1:
  greedy walk to nearest neighbor
  maintain candidate heap of size ef_search
return k best from final layer heap
```

#### Vector Databases Using HNSW

**FAISS (Facebook AI Similarity Search):**
- Library by Facebook
- Multiple index types including HNSW
- GPU support
- Python interface
- In-memory or disk-based

**Milvus:**
- Open-source vector database
- Supports HNSW and other indexes
- Distributed architecture
- Cloud-native
- Good for production

**Weaviate:**
- Open-source vector database
- GraphQL API
- Built-in vectorization
- Multi-tenancy support
- Good for semantic search

**ChromaDB:**
- Lightweight vector database
- Simple Python API
- Good for development and small deployments
- In-memory or persistent
- Easy integration with LangChain

**Pinecone:**
- Managed vector database
- Serverless architecture
- Auto-scaling
- Pay-as-you-go
- Good for production without infrastructure management

#### Comparison of Vector Databases

| Feature | FAISS | Milvus | Weaviate | ChromaDB | Pinecone |
|---------|-------|--------|----------|----------|----------|
| Type | Library | Database | Database | Database | Managed |
| HNSW | Yes | Yes | Yes | Yes | Yes |
| Scalability | High | Very High | High | Medium | Very High |
| Ease of Use | Medium | Medium | High | Very High | Very High |
| Cost | Free | Free | Free | Free | Paid |
| Production Ready | Yes | Yes | Yes | Medium | Yes |

### Readings

- BM25 algorithm papers:
  - "Okapi at TREC-3" (Robertson et al., 1995)
  - "The Probabilistic Relevance Framework: BM25 and Beyond" (Robertson & Zaragoza, 2009)

- HNSW and approximate nearest neighbor search papers:
  - "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2016)
  - "Revisiting Approximate Nearest Neighbor Search" (survey papers)

 

### Additional Resources

- [BM25 Python Implementation](https://github.com/dorianbrown/rank_bm25)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Milvus Documentation](https://milvus.io/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)

### Practical Code Examples

#### Hybrid Search Implementation

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HybridSearch:
    def __init__(self, documents: List[str], alpha=0.7):
        self.documents = documents
        self.alpha = alpha  # Weight for semantic search
        
        # Initialize BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Initialize embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_embeddings = self.embedding_model.encode(documents)
    
    def search(self, query: str, top_k: int = 10):
        """Hybrid search combining BM25 and semantic search"""
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        
        # Semantic search scores
        query_embedding = self.embedding_model.encode(query)
        semantic_scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # Combine scores
        hybrid_scores = self.alpha * semantic_scores + (1 - self.alpha) * bm25_scores
        
        # Get top-k
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = [
            (i, hybrid_scores[i], self.documents[i])
            for i in top_indices
        ]
        
        return results

# Usage
docs = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing analyzes text",
    "Computer vision processes images"
]

searcher = HybridSearch(docs, alpha=0.7)
results = searcher.search("AI and neural networks", top_k=3)

for idx, score, doc in results:
    print(f"Score: {score:.3f} - {doc}")
```

#### HNSW with FAISS

```python
import faiss
import numpy as np

class HNSWIndex:
    def __init__(self, dimension: int, M=16, ef_construction=200):
        self.dimension = dimension
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
    
    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to index"""
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
    
    def search(self, query_vector: np.ndarray, k: int = 10, ef_search: int = 50):
        """Search for k nearest neighbors"""
        # Normalize query vector
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Set ef_search parameter
        self.index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        return indices[0], distances[0]

# Usage
dimension = 384
index = HNSWIndex(dimension, M=16, ef_construction=200)

# Add vectors (example: 1000 random vectors)
vectors = np.random.random((1000, dimension)).astype('float32')
index.add_vectors(vectors)

# Search
query = np.random.random((1, dimension)).astype('float32')
indices, distances = index.search(query, k=5, ef_search=50)

print(f"Top 5 results: {indices}")
print(f"Distances: {distances}")
```

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Low BM25 scores** | All documents score near zero | Check tokenization, verify documents contain query terms, adjust k1 parameter |
| **Poor hybrid search results** | Hybrid performs worse than individual methods | Adjust alpha weight, normalize scores properly, check score ranges |
| **Slow HNSW search** | High query latency | Reduce ef_search, increase M for faster build, use quantization |
| **High memory usage** | Vector index consumes too much RAM | Use quantization (PQ/SQ), reduce M parameter, consider sharding |
| **Low recall** | Missing relevant documents | Increase ef_search, adjust M parameter, use reranking |
| **Inconsistent results** | Different results for same query | Check normalization, ensure deterministic seed, verify vector preprocessing |

**Common Pitfalls:**
- **Pitfall:** Not normalizing BM25 and semantic scores before combining
  - **Solution:** Always normalize both score types to [0, 1] range before weighted combination
- **Pitfall:** Using default HNSW parameters without tuning
  - **Solution:** Tune M and ef_search based on your recall/latency requirements
- **Pitfall:** Building index without considering update patterns
  - **Solution:** Use appropriate index type (static vs. dynamic) based on update frequency

### Quick Reference Guide

#### Search Method Selection Matrix

| Use Case | Recommended Method | Why |
|----------|------------------|-----|
| Exact keyword matching | BM25 | Fast, precise for keywords |
| Semantic understanding | Embeddings | Handles synonyms, paraphrasing |
| Production RAG | Hybrid (BM25 + Embeddings) | Best overall performance |
| Large-scale search | HNSW | Fast approximate search |
| Real-time updates | Dynamic HNSW | Supports incremental updates |
| Cost-sensitive | BM25 | Lower computational cost |

#### BM25 Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| k1 | 0.5 - 2.0 | Higher = more TF emphasis |
| b | 0.0 - 1.0 | Higher = more length normalization |

#### Vector Database Selection

| Requirement | Recommended | Alternatives |
|-------------|-------------|--------------|
| Development/Prototyping | ChromaDB | FAISS |
| Production (self-hosted) | Milvus | Weaviate |
| Managed service | Pinecone | Weaviate Cloud |
| GPU acceleration | FAISS | Milvus |
| Multi-tenancy | Weaviate | Milvus |

### Case Studies

#### Case Study: E-commerce Search Optimization

**Challenge:** An e-commerce platform needed to improve product search accuracy and handle synonyms.

**Initial Setup:**
- Pure keyword search (SQL LIKE)
- 45% relevant results
- No synonym handling
- Fast but inaccurate

**Solution:**
- Implemented hybrid search (BM25 + embeddings)
- Added query expansion for product synonyms
- Used HNSW for fast vector search

**Results:**
- 82% relevant results (45% → 82%)
- 30% improvement in conversion rate
- 2x faster search with HNSW

**Lessons Learned:**
- Hybrid search critical for product search
- Query expansion improved recall significantly
- HNSW enabled real-time search at scale

### Hands-On Lab: Build a Hybrid Search System

**Lab Objective:** Implement a complete hybrid search system combining BM25 and semantic search.

**Steps:**

1. **Setup**
```bash
pip install rank-bm25 sentence-transformers faiss-cpu numpy scikit-learn
```

2. **Implement Components**
```python
# Use code examples above
# Implement BM25 search
# Implement semantic search
# Combine with weighted scores
```

3. **Evaluate Performance**
```python
# Test on sample dataset
# Measure recall@k
# Compare individual vs hybrid performance
```

4. **Optimize Parameters**
```python
# Tune BM25 parameters (k1, b)
# Tune hybrid weights (alpha)
# Optimize HNSW parameters
```

**Expected Outcomes:**
- Working hybrid search system
- Understanding of parameter tuning
- Knowledge of performance trade-offs
- Ability to optimize for specific use cases

### Testing Examples

#### Unit Test for BM25

```python
import unittest
from your_module import BM25

class TestBM25(unittest.TestCase):
    def setUp(self):
        self.documents = [
            "machine learning is important",
            "deep learning uses neural networks",
            "machine learning algorithms learn from data"
        ]
        self.bm25 = BM25(self.documents)
    
    def test_search_returns_results(self):
        """Test that search returns expected number of results"""
        results = self.bm25.search("machine learning", top_k=2)
        self.assertEqual(len(results), 2)
    
    def test_relevant_document_ranked_higher(self):
        """Test that relevant documents score higher"""
        results = self.bm25.search("neural networks", top_k=3)
        # Document about neural networks should be in top results
        doc_indices = [idx for idx, _ in results]
        self.assertIn(1, doc_indices[:2])  # Index 1 contains "neural networks"
    
    def test_idf_calculation(self):
        """Test IDF calculation"""
        # Common terms should have lower IDF
        common_idf = self.bm25.idf_scores.get("learning", 0)
        rare_idf = self.bm25.idf_scores.get("neural", 0)
        self.assertLess(common_idf, rare_idf)

if __name__ == "__main__":
    unittest.main()
```

### Glossary

**BM25 (Best Matching 25):** A probabilistic ranking function that improves upon TF-IDF by handling term frequency saturation and document length normalization.

**TF-IDF (Term Frequency-Inverse Document Frequency):** A statistical measure that evaluates word importance by combining term frequency with inverse document frequency.

**Sparse Retrieval:** Search method using sparse vectors (mostly zeros) where each dimension represents a word/token, examples include TF-IDF and BM25.

**Dense Retrieval:** Search method using dense, continuous-valued vectors (embeddings) that capture semantic meaning.

**Hybrid Search:** Combining multiple retrieval methods (typically BM25 and embeddings) to leverage strengths of both approaches.

**HNSW (Hierarchical Navigable Small World):** A graph-based algorithm for fast approximate nearest neighbor search with logarithmic complexity.

**ANN (Approximate Nearest Neighbor):** Search algorithms that find approximate nearest neighbors faster than exact search, trading some accuracy for speed.

**Recall@k:** Evaluation metric measuring the proportion of relevant documents found in the top-k results.

**NDCG (Normalized Discounted Cumulative Gain):** Evaluation metric that measures ranking quality by considering both relevance and position.

**Reranking:** A second-stage process that re-orders retrieved documents using a more expensive but accurate model.

### Key Takeaways

1. Keyword search (BM25) and semantic search (embeddings) serve different purposes
2. BM25 is excellent for exact keyword matching and fast retrieval
3. Embedding search excels at semantic understanding and synonyms
4. Hybrid approaches often provide best results
5. HNSW enables fast approximate nearest neighbor search at scale
6. Vector database choice depends on scale, requirements, and budget
7. Parameter tuning is crucial for optimal search performance
8. Always normalize scores before combining in hybrid search
9. Trade-offs exist between recall, latency, and memory usage
10. Testing and evaluation are essential for production systems

---

**Previous Module:** [Module 3: Representations & Search Algorithms](../module_03.md)  
**Next Module:** [Module 5: Frameworks for Building GenAI Applications](../module_05.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

