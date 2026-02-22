# 🔄 RAG Workflow: How VectorAI DB Fits In

This document explains how **Actian VectorAI DB** integrates into a Retrieval-Augmented Generation (RAG) application.

## 📊 The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG APPLICATION WORKFLOW                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: INDEXING (One-time setup)                             │
└─────────────────────────────────────────────────────────────────┘

   📄 Your Documents
   (PDF, txt, web pages, etc.)
          │
          ▼
   ┌─────────────────┐
   │  Chunk Text     │  Split into manageable pieces
   │  200-500 chars  │  with overlap for context
   └────────┬────────┘
          │
          ▼
   ┌──────────────────┐
   │ Embedding Model  │  Convert text → vectors
   │ (sentence-       │  e.g., "database" → [0.2, 0.8, -0.3, ...]
   │  transformers)   │
   └────────┬─────────┘
          │
          ▼
   ╔═══════════════════╗
   ║  VectorAI DB      ║  ← Store vectors + metadata
   ║  (Actian)         ║     Fast similarity search
   ╚═══════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: RETRIEVAL (Every user query)                          │
└─────────────────────────────────────────────────────────────────┘

   👤 User Question
   "What are the key features?"
          │
          ▼
   ┌──────────────────┐
   │ Embedding Model  │  Convert query → vector
   │ (same as above)  │  [0.15, 0.75, -0.25, ...]
   └────────┬─────────┘
          │
          ▼
   ╔═══════════════════╗
   ║  VectorAI DB      ║  Search for similar vectors
   ║  .search()        ║  Returns top K matches
   ╚═══════╦═══════════╝
          │
          ▼
   ┌──────────────────┐
   │ Retrieved Chunks │  Get the actual text
   │ • Chunk A (0.89) │  from similar vectors
   │ • Chunk B (0.85) │
   │ • Chunk C (0.78) │
   └────────┬─────────┘
          │
          ▼
   ┌──────────────────┐
   │  LLM (GPT, etc.) │  Generate answer using
   │  + Context       │  the retrieved context
   └────────┬─────────┘
          │
          ▼
   ✅ Generated Answer
   "The key features are..."
```

## 🔍 Why Each Component?

### 1. Chunking
**Problem:** Documents are too long for embedding models (max 512 tokens usually).  
**Solution:** Split into smaller chunks that fit the model's context window.

### 2. Embedding Model
**Problem:** Can't compare text directly ("car" vs "automobile" should match).  
**Solution:** Convert text to vectors in high-dimensional space where similar meanings are close together.

### 3. VectorAI DB
**Problem:** Need to quickly find most similar vectors from millions of possibilities.  
**Solution:** HNSW algorithm enables fast approximate nearest neighbor search.

### 4. LLM
**Problem:** Retrieved chunks are just raw text - need coherent, natural answer.  
**Solution:** LLM synthesizes retrieved context into a well-formed response.

## 🎯 Real Example

Let's walk through a real query:

### Query: "How do I start the database?"

**Step 1: Embed the query**
```python
query_vector = model.encode("How do I start the database?")
# Result: [0.12, 0.67, -0.34, 0.89, ..., 0.23]  (384 dimensions)
```

**Step 2: Search VectorAI DB**
```python
results = client.search(
    collection="docs",
    query=query_vector,
    top_k=3
)
```

**Step 3: VectorAI DB returns similar vectors**
```
Match 1 (score: 0.89):
  "To get started with Actian VectorAI DB, you need to first pull 
   the Docker image and start the container. Use docker compose up..."

Match 2 (score: 0.82):
  "The database runs on port 50051 by default. You can use 
   docker compose up to start it easily..."

Match 3 (score: 0.78):
  "Make sure you have Docker installed. Start the database with
   docker compose up -d for background mode..."
```

**Step 4: Send to LLM with context**
```
System: Answer based on this context: [retrieved chunks]
User: How do I start the database?
```

**Step 5: Get generated answer**
```
"To start the Actian VectorAI DB, first ensure Docker is installed,
then run `docker compose up` to start the database. It will be 
available on port 50051. You can add `-d` flag to run in background."
```

## 🚫 Without Vector DB (Traditional Search)

```
Query: "How do I start the database?"

Traditional keyword search:
  ❌ Looks for exact words: "start" AND "database"
  ❌ Misses: "begin", "launch", "initialize"
  ❌ Misses: "DB", "vector store"
  ❌ Returns: Everything with those keywords (not ranked by relevance)
```

## ✅ With Vector DB (Semantic Search)

```
Query: "How do I start the database?"

Semantic search:
  ✅ Understands intent: "starting/launching/initializing"
  ✅ Understands context: "database/DB/vector store"
  ✅ Returns: Most semantically similar content
  ✅ Ranked: By similarity score (0.0 to 1.0)
```

## 🔢 The Math Behind It

### Cosine Similarity

VectorAI DB compares vectors using cosine similarity:

```
                    A · B
similarity = ─────────────────
             ||A|| × ||B||

Where:
  • A = query vector
  • B = stored document vector
  • Result = -1 to 1 (1 = identical, 0 = unrelated, -1 = opposite)
```

**Example:**
```python
query:    [0.8, 0.6, 0.0]
chunk_1:  [0.9, 0.5, 0.1]  → similarity: 0.89  ← Very similar!
chunk_2:  [0.2, 0.3, 0.8]  → similarity: 0.34  ← Less similar
```

## 🎓 Key Concepts

### Embeddings
> Numerical representations of text that capture semantic meaning.

### Vector Database
> Specialized database optimized for storing and searching high-dimensional vectors.

### Semantic Search
> Finding information based on meaning, not just keywords.

### HNSW (Hierarchical Navigable Small World)
> Graph-based algorithm for fast approximate nearest neighbor search.

### Retrieval-Augmented Generation (RAG)
> Using retrieved context to help LLMs generate more accurate, grounded answers.

## 💡 Why This Architecture?

### ✅ Advantages

1. **Accurate:** LLM answers are grounded in your actual documents
2. **Current:** Update documents without retraining the LLM
3. **Transparent:** You know exactly what context was used
4. **Cost-effective:** Don't need to fine-tune expensive models
5. **Scalable:** VectorAI DB handles millions of documents efficiently

### 🤔 When to Use RAG vs. Fine-tuning

**Use RAG when:**
- You have frequently changing information
- You need to cite sources
- You want to control what the LLM knows
- You have domain-specific documents

**Use Fine-tuning when:**
- You need to change the model's style/tone
- You want the model to learn specific patterns
- You have static knowledge
- You need the model to internalize knowledge

## 🛠️ Integration Points

Where does VectorAI DB fit in your stack?

```
Your Application Stack:

┌─────────────────────┐
│   Frontend (UI)     │  User asks questions
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Backend API       │  Orchestrates the flow
│   (Python/Node/Go)  │
└──────────┬──────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
┌────────────┐  ┌────────────┐
│ VectorAI   │  │ LLM API    │
│ DB         │  │ (OpenAI,   │
│ (Actian)   │  │  Claude)   │
└────────────┘  └────────────┘
    ▲
    │
┌───┴──────────────┐
│ Embedding Model  │
│ (sentence-trans) │
└──────────────────┘
```

## 🎯 Summary

**VectorAI DB is your "smart memory"** - it stores your documents as vectors and quickly finds the most relevant ones for any query. Combined with an LLM, you get accurate, contextual answers grounded in your actual data.

**The RAG pattern is:**
1. Index your docs once
2. For each query: retrieve → generate → respond
3. Update docs anytime without retraining

---

**Ready to build?** Check out the [RAG example](rag_example.py) to see this in action!
