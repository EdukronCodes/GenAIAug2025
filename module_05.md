# Module 5: Frameworks for Building GenAI Applications

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Classes:** 8-9

---

## Class 8: LangChain – The Core Framework

### Topics Covered

- Core components: Chains, Agents, Tools, Memory
- Building a simple RAG pipeline
- LangChain Expression Language (LCEL)
- Integration with OpenAI and ChromaDB

### Learning Objectives

By the end of this class, students will be able to:
- Understand LangChain's architecture and core concepts
- Build chains for sequential processing
- Create agents with tool use capabilities
- Implement memory in conversational applications
- Build a complete RAG pipeline using LangChain

### Core Concepts

#### LangChain Overview

**What is LangChain?**
- Framework for developing LLM-powered applications
- Modular components that can be combined
- Abstraction layer over different LLMs
- Integration with various tools and databases

**Key Benefits:**
- Rapid prototyping
- Modular architecture
- Extensive integrations
- Active community
- Production-ready features

#### Core Components

**1. Chains**
- Sequences of operations
- Combine multiple components
- Examples: LLMChain, SequentialChain, RouterChain

**2. Agents**
- Autonomous systems that use tools
- Make decisions based on observations
- Examples: ReAct, Plan-and-Execute, Self-Ask-with-Search

**3. Tools**
- External functions agents can use
- Examples: Web search, calculator, database queries
- Custom tools can be created

**4. Memory**
- Conversation history management
- Types: ConversationBufferMemory, ConversationSummaryMemory
- Enables context-aware conversations

**5. Prompts**
- Prompt templates and management
- Few-shot examples
- Prompt versioning

**6. Document Loaders**
- Load documents from various sources
- Examples: PDF, CSV, web pages, databases

**7. Text Splitters**
- Chunk documents appropriately
- Respect token limits
- Overlap strategies

**8. Vector Stores**
- Integration with vector databases
- Embedding and retrieval
- Examples: ChromaDB, Pinecone, FAISS

**9. Retrievers**
- Document retrieval strategies
- Types: Vector store retriever, BM25 retriever, Hybrid

**10. Output Parsers**
- Structure LLM outputs
- Validate and parse responses
- Examples: Pydantic, JSON

#### Building a Simple RAG Pipeline

**Step-by-Step RAG with LangChain:**

```python
# 1. Load documents
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# 5. Create LLM
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)

# 6. Create prompt template
from langchain.prompts import PromptTemplate
template = """Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 7. Create chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 8. Query
response = qa_chain({"query": "What is the main topic?"})
```

#### LangChain Expression Language (LCEL)

**What is LCEL?**
- Declarative way to compose chains
- Type-safe
- Better streaming support
- Easier debugging

**Example:**
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
result = chain.invoke({"topic": "AI"})
```

**Benefits:**
- Cleaner syntax
- Better composability
- Built-in streaming
- Parallel execution support

#### Integration Examples

**OpenAI Integration:**
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Completion model
llm = OpenAI(model_name="text-davinci-003")

# Chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo")
```

**ChromaDB Integration:**
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

---

## Class 9: LlamaIndex & Other Frameworks

### Topics Covered

- Overview of LlamaIndex, differences from LangChain
- Index types: Summary, Vector, List
- Integration with LLM APIs and databases
- When to use LlamaIndex vs LangChain

### Learning Objectives

By the end of this class, students will be able to:
- Understand LlamaIndex architecture and philosophy
- Create different types of indexes
- Compare LlamaIndex with LangChain
- Choose appropriate framework for use case
- Build applications using LlamaIndex

### Core Concepts

#### LlamaIndex Overview

**What is LlamaIndex?**
- Data framework for LLM applications
- Focus on data ingestion and indexing
- Query interface for structured/unstructured data
- Built for RAG and knowledge base applications

**Key Philosophy:**
- Data-first approach
- Flexible indexing strategies
- Query optimization
- Production-ready data pipelines

#### LlamaIndex vs LangChain

**LlamaIndex:**
- Focus: Data ingestion and indexing
- Strength: RAG, knowledge bases, data queries
- Best for: Document-heavy applications
- Query interface: Rich query capabilities

**LangChain:**
- Focus: Application orchestration
- Strength: Chains, agents, tool use
- Best for: Complex workflows, agentic systems
- Flexibility: More modular components

**When to Use LlamaIndex:**
- Document-heavy applications
- Knowledge base queries
- Need for advanced indexing strategies
- Data-centric workflows

**When to Use LangChain:**
- Complex agent workflows
- Multiple tool integrations
- Chain compositions
- Application orchestration

**Can Use Both:**
- LlamaIndex for data layer
- LangChain for application layer
- Complementary strengths

#### Index Types

**1. Vector Store Index**
- Most common type
- Stores embeddings in vector database
- Semantic search capability
- Good for general RAG

**2. Summary Index**
- Creates summaries of documents
- Hierarchical summaries
- Good for overview queries
- Efficient for large documents

**3. List Index**
- Sequential document list
- Simple retrieval
- Good for small datasets
- Deterministic ordering

**4. Tree Index**
- Hierarchical tree structure
- Top-down querying
- Good for structured documents
- Efficient for specific queries

**5. Keyword Table Index**
- Keyword-based indexing
- Fast keyword lookup
- Good for exact matches
- Complementary to vector index

#### Building with LlamaIndex

**Basic RAG Example:**
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What is the main topic?")
print(response)
```

**Advanced Example with Multiple Indexes:**
```python
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage
)

# Create vector index
vector_index = VectorStoreIndex.from_documents(documents)

# Create summary index
summary_index = SummaryIndex.from_documents(documents)

# Query both
vector_response = vector_index.as_query_engine().query(question)
summary_response = summary_index.as_query_engine().query(question)
```

#### Integration with LLM APIs

**OpenAI:**
```python
from llama_index.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)
```

**Anthropic:**
```python
from llama_index.llms import Anthropic

llm = Anthropic(model="claude-3-sonnet-20240229")
```

**Local Models:**
```python
from llama_index.llms import Ollama

llm = Ollama(model="llama2")
```

#### Integration with Databases

**ChromaDB:**
```python
from llama_index.vector_stores import ChromaVectorStore
import chromadb

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

**Pinecone:**
```python
from llama_index.vector_stores import PineconeVectorStore
import pinecone

pinecone.init(api_key="your-key")
index = pinecone.Index("my-index")
vector_store = PineconeVectorStore(pinecone_index=index)
```

#### Other Frameworks

**Haystack (by deepset):**
- Focus: Production-ready NLP pipelines
- Strength: Document processing, QA systems
- Good for: Enterprise applications

**Semantic Kernel (Microsoft):**
- Focus: AI orchestration for .NET
- Strength: Enterprise integration
- Good for: Microsoft ecosystem

**AutoGPT / LangGraph:**
- Focus: Agentic workflows
- Strength: Multi-step reasoning
- Good for: Complex autonomous agents

### Readings

- LangChain documentation: https://python.langchain.com/
- LlamaIndex documentation: https://docs.llamaindex.ai/
- Framework comparison articles

 

### Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex GitHub](https://github.com/run-llama/llama_index)
- [Framework Comparison Guide](https://www.pinecone.io/learn/langchain-vs-llamaindex/)

### Practical Code Examples

#### Complete LangChain RAG Pipeline

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class LangChainRAG:
    def __init__(self, documents_path, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0)
        self.persist_directory = persist_directory
        self._setup_vectorstore(documents_path)
    
    def _setup_vectorstore(self, documents_path):
        """Load documents and create vector store"""
        loader = PyPDFLoader(documents_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def create_qa_chain(self):
        """Create QA chain with custom prompt"""
        prompt_template = """Use the following pieces of context to answer the question.
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def query(self, question):
        """Query the RAG system"""
        qa_chain = self.create_qa_chain()
        result = qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result.get("source_documents", [])
        }

# Usage
rag = LangChainRAG("./documents.pdf")
response = rag.query("What is the main topic?")
print(response["answer"])
```

**Pro Tip:** Use LCEL (LangChain Expression Language) for cleaner, more maintainable chain definitions. It provides better debugging and streaming capabilities.

**Common Pitfall:** Not properly managing document chunking can lead to information loss. Always review chunk boundaries and adjust overlap based on document structure.

### Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Import errors** | ModuleNotFoundError | Install missing packages: `pip install langchain openai chromadb` |
| **Vector store errors** | Persistence failures | Check directory permissions, ensure path exists |
| **Empty retrieval** | No documents returned | Verify vector store has data, check embedding model, adjust similarity threshold |
| **Slow performance** | High latency | Use caching, optimize chunk size, batch operations |
| **Memory issues** | Out of memory | Reduce batch size, use streaming, optimize vector store |

### Quick Reference Guide

#### Framework Selection Matrix

| Use Case | Recommended Framework | Why |
|----------|---------------------|-----|
| Application orchestration | LangChain | Better agent workflows, tool integration |
| Data ingestion | LlamaIndex | Optimized loaders, indexing |
| RAG systems | Both | LangChain for chains, LlamaIndex for indexing |
| Enterprise apps | LangChain | Better integration, production features |
| Knowledge bases | LlamaIndex | Specialized query engines |

### Case Studies

#### Case Study: Building Production RAG with LangChain

**Challenge:** A company needed a production-ready RAG system for customer support.

**Solution:** Built with LangChain using:
- ChromaDB for vector storage
- GPT-4 for generation
- Custom prompt templates
- Error handling and retries

**Results:**
- 85% accuracy
- 2-second response time
- 70% cost reduction

**Lessons Learned:**
- LangChain's modularity enabled rapid iteration
- Proper error handling critical for production
- Monitoring and observability essential

### Key Takeaways

1. LangChain excels at application orchestration and agent workflows
2. LlamaIndex is optimized for data ingestion and knowledge base queries
3. Both frameworks can be used together for complementary strengths
4. LCEL provides a clean, declarative way to compose chains
5. Framework choice depends on use case and requirements
6. Integration with various LLMs and databases is straightforward in both
7. Production considerations (scaling, monitoring) should guide framework selection
8. Error handling and retries are essential for production systems
9. Proper chunking and retrieval configuration significantly impact performance
10. Testing and validation should be integrated into development workflow

---

**Previous Module:** [Module 4: Search Algorithms & Retrieval Techniques](../module_04.md)  
**Next Module:** [Module 6: RAG & Transformer Architecture](../module_06.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

