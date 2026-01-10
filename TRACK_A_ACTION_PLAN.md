# 🎯 Track A: Action Plan for KDSH 2026

## 📋 Problem Summary (What You Need to Do)

### The Core Task
You are building a **binary classification system** that determines:

> **Given a full novel (100k+ words) and a hypothetical backstory for a character, is the backstory CONSISTENT (1) or INCONSISTENT (0) with the novel?**

### What You're Given
1. **A complete novel** - Full text, 100k+ words, no truncation
2. **A hypothetical backstory** - Newly written character background describing:
   - Early-life events
   - Formative experiences
   - Beliefs, fears, ambitions
   - Assumptions about the world

### What You Must Output
1. **Binary Label**: `1` (Consistent) or `0` (Inconsistent)
2. **Evidence Rationale** (Optional but encouraged): Explain why with text excerpts

---

## 🧠 Key Concepts You Need to Learn

### Tier 1: Essential (Learn These First)
| Concept | Description | Why It's Important |
|---------|-------------|-------------------|
| **RAG (Retrieval-Augmented Generation)** | Technique to retrieve relevant documents before generating answers | Core of your pipeline - retrieve relevant novel sections |
| **Vector Embeddings** | Converting text to numerical vectors for semantic search | How you'll search through 100k+ words efficiently |
| **Chunking Strategies** | Breaking large documents into smaller pieces | Novels are huge; you need smart splitting |
| **Semantic Search** | Finding similar text by meaning, not keywords | Finding relevant evidence in the novel |
| **LLM Prompting** | Crafting instructions for language models | How you'll get the model to reason about consistency |
| **Pathway Framework** | Python framework for data processing (REQUIRED) | Mandatory for Track A submissions |

### Tier 2: Important for Quality
| Concept | Description | Why It's Important |
|---------|-------------|-------------------|
| **Reranking** | Re-scoring retrieved documents for relevance | Improves retrieval precision |
| **Chain-of-Thought Prompting** | Making LLMs show reasoning steps | Better causal reasoning |
| **Long Context Handling** | Strategies for processing documents > context window | Novels exceed most model limits |
| **Causal Reasoning** | Determining cause-effect relationships | Core task requirement |
| **Cross-Encoder Models** | Models that compare two texts directly | Better relevance scoring |

### Tier 3: Advanced Optimization
| Concept | Description | Why It's Important |
|---------|-------------|-------------------|
| **Hierarchical Summarization** | Multi-level document summaries | Preserving global coherence |
| **Hybrid Search** | Combining keyword + semantic search | Better recall |
| **Agentic Pipelines** | Multi-step LLM workflows | Complex reasoning chains |
| **Memory Mechanisms** | Persistent state across retrievals | Tracking accumulating constraints |

---

## 🏗️ Phase 1: Minimum Viable Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1 PIPELINE                             │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   Novel      │     │   Backstory  │     │  Pathway     │
  │   (.txt)     │     │   (text)     │     │  Ingestion   │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         ▼                    │                    │
  ┌──────────────┐            │                    │
  │   Chunking   │◄───────────┼────────────────────┘
  │   Strategy   │            │
  └──────┬───────┘            │
         │                    │
         ▼                    │
  ┌──────────────┐            │
  │  Embedding   │            │
  │  Generation  │            │
  └──────┬───────┘            │
         │                    │
         ▼                    │
  ┌──────────────┐     ┌──────┴───────┐
  │   Vector     │     │   Query      │
  │   Store      │◄────┤   Formation  │
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐
  │  Retrieval   │
  │  (Top-K)     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  LLM Prompt  │
  │  + Reasoning │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Output:    │
  │   0 or 1     │
  │  + Rationale │
  └──────────────┘
```

### Step-by-Step Implementation

#### Step 1: Project Setup with Pathway
```python
# requirements.txt
pathway
openai  # or any LLM provider
sentence-transformers
chromadb  # or pathway's built-in vector store
python-dotenv
```

```bash
pip install pathway openai sentence-transformers chromadb python-dotenv
```

#### Step 2: Data Ingestion with Pathway
```python
# ingest.py
import pathway as pw
from pathway.xpacks.llm import embedders

# Read novels from folder (Pathway requirement)
class NovelSchema(pw.Schema):
    text: str
    story_id: str

# Use Pathway's file connector
novels = pw.io.fs.read(
    path="./data/novels/",
    format="plaintext",
    mode="static"
)
```

#### Step 3: Chunking Strategy
```python
# chunker.py
def chunk_novel(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Semantic-aware chunking with overlap.
    
    Args:
        text: Full novel text
        chunk_size: Target words per chunk
        overlap: Words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # Overlap for context continuity
    
    return chunks
```

#### Step 4: Vector Store Setup
```python
# vector_store.py
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good baseline

# Create vector store
client = chromadb.Client()
collection = client.create_collection("novels")

def index_novel(story_id: str, chunks: list[str]):
    """Index novel chunks into vector store."""
    embeddings = embedder.encode(chunks).tolist()
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{story_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"story_id": story_id, "chunk_idx": i} for i in range(len(chunks))]
    )
```

#### Step 5: Retrieval
```python
# retrieval.py
def retrieve_evidence(backstory: str, story_id: str, top_k: int = 10) -> list[str]:
    """
    Retrieve relevant novel chunks for a backstory.
    
    Args:
        backstory: The hypothetical character backstory
        story_id: ID of the novel to search
        top_k: Number of chunks to retrieve
    
    Returns:
        List of relevant text chunks
    """
    # Embed the backstory
    query_embedding = embedder.encode([backstory]).tolist()
    
    # Query vector store
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"story_id": story_id}
    )
    
    return results['documents'][0]
```

#### Step 6: LLM Reasoning
```python
# reasoning.py
import openai

def classify_consistency(backstory: str, evidence: list[str]) -> tuple[int, str]:
    """
    Use LLM to classify consistency.
    
    Args:
        backstory: The hypothetical character backstory
        evidence: Retrieved novel excerpts
    
    Returns:
        (prediction, rationale)
    """
    evidence_text = "\n\n---\n\n".join(evidence)
    
    prompt = f"""You are analyzing whether a character backstory is CONSISTENT or INCONSISTENT with a novel.

## Character Backstory:
{backstory}

## Relevant Excerpts from the Novel:
{evidence_text}

## Your Task:
1. Analyze whether the backstory's claims align with the novel's events, character development, and constraints.
2. Look for:
   - Direct contradictions (explicit conflicts)
   - Causal impossibilities (the backstory makes later events impossible)
   - Character inconsistencies (the backstory doesn't fit the character's established traits)
   - Temporal conflicts (timeline doesn't work)

## Output Format:
First, provide your reasoning step by step.
Then, provide your final answer in this exact format:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [One or two sentences explaining your decision]
"""

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Lower temperature for more consistent reasoning
    )
    
    output = response.choices[0].message.content
    
    # Parse output
    prediction = 1 if "PREDICTION: 1" in output else 0
    rationale = output.split("RATIONALE:")[-1].strip() if "RATIONALE:" in output else ""
    
    return prediction, rationale
```

#### Step 7: Main Pipeline
```python
# main.py
import csv

def process_dataset(input_path: str, output_path: str):
    """Main pipeline to process all examples."""
    
    results = []
    
    # Load test examples (you'll need to adapt this to actual dataset format)
    examples = load_examples(input_path)
    
    for example in examples:
        story_id = example['story_id']
        backstory = example['backstory']
        novel_path = example['novel_path']
        
        # Step 1: Load and chunk novel
        novel_text = load_novel(novel_path)
        chunks = chunk_novel(novel_text)
        
        # Step 2: Index chunks (once per novel)
        index_novel(story_id, chunks)
        
        # Step 3: Retrieve evidence
        evidence = retrieve_evidence(backstory, story_id)
        
        # Step 4: Classify
        prediction, rationale = classify_consistency(backstory, evidence)
        
        results.append({
            'Story ID': story_id,
            'Prediction': prediction,
            'Rationale': rationale
        })
    
    # Write results
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Story ID', 'Prediction', 'Rationale'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    process_dataset("./data/test.json", "./results.csv")
```

---

## 🚀 Phase 2: Improvements & Fine-tuning

### 1. Better Chunking Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Semantic Chunking** | Split at paragraph/scene boundaries | Better context preservation |
| **Character-Aware Chunking** | Chunk around character mentions | Better for character consistency |
| **Hierarchical Chunking** | Multiple granularity levels | Complex evidence gathering |

```python
# Advanced: Semantic chunking with spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunk(text: str, max_chunk_size: int = 1500) -> list[str]:
    """Split text at sentence boundaries, respecting max size."""
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sent in doc.sents:
        sent_len = len(sent.text.split())
        if current_size + sent_len > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sent.text]
            current_size = sent_len
        else:
            current_chunk.append(sent.text)
            current_size += sent_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### 2. Better Embeddings

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| `all-MiniLM-L6-v2` | Medium | Fast | Good baseline |
| `all-mpnet-base-v2` | High | Medium | Better accuracy |
| `text-embedding-3-small` | High | API | OpenAI hosted |
| `text-embedding-3-large` | Highest | API | Best quality |
| `BGE-large-en` | High | Medium | Open source, excellent |

### 3. Reranking for Better Retrieval

```python
# reranker.py
from sentence_transformers import CrossEncoder

# Cross-encoder reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_evidence(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """Rerank retrieved documents for better precision."""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score and return top_k
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for score, doc in ranked[:top_k]]
```

### 4. Multi-Query Retrieval

```python
def multi_query_retrieve(backstory: str, story_id: str) -> list[str]:
    """
    Generate multiple queries from backstory for better coverage.
    Extract key claims and search for each.
    """
    # Extract key claims from backstory using LLM
    claims_prompt = f"""Extract the key claims from this character backstory as a list:

{backstory}

Output each claim on a new line, starting with a dash."""

    claims_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": claims_prompt}]
    )
    
    claims = claims_response.choices[0].message.content.split("\n")
    claims = [c.strip("- ") for c in claims if c.strip()]
    
    # Retrieve for each claim
    all_evidence = []
    for claim in claims:
        evidence = retrieve_evidence(claim, story_id, top_k=3)
        all_evidence.extend(evidence)
    
    # Deduplicate and rerank
    unique_evidence = list(set(all_evidence))
    return rerank_evidence(backstory, unique_evidence)
```

### 5. Hierarchical Summarization for Long Context

```python
def create_novel_summary(chunks: list[str]) -> str:
    """Create a hierarchical summary of the novel."""
    # First, summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)
    
    # Then, create an overall summary
    combined = "\n\n".join(chunk_summaries)
    overall_summary = summarize_chunk(combined)
    
    return overall_summary

def summarize_chunk(text: str) -> str:
    """Summarize a single chunk."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user", 
            "content": f"Summarize this text in 2-3 sentences, focusing on key events, character traits, and plot points:\n\n{text}"
        }]
    )
    return response.choices[0].message.content
```

### 6. Chain-of-Thought Reasoning

```python
def classify_with_cot(backstory: str, evidence: list[str]) -> tuple[int, str]:
    """Use Chain-of-Thought for better reasoning."""
    
    evidence_text = "\n\n---\n\n".join(evidence)
    
    prompt = f"""You are a careful literary analyst. Analyze whether this character backstory is consistent with the novel.

## Backstory:
{backstory}

## Novel Excerpts:
{evidence_text}

## Step-by-Step Analysis:

### Step 1: List Key Claims
What are the main claims in the backstory? List each one.

### Step 2: Evidence Mapping
For each claim, what evidence supports or contradicts it?

### Step 3: Causal Analysis
Do the backstory events causally enable or prevent the novel's events?

### Step 4: Character Consistency
Does the backstory match the character's established personality/traits?

### Step 5: Final Judgment
Based on the above analysis:
PREDICTION: [1 or 0]
RATIONALE: [summary]
"""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    # Parse and return
    output = response.choices[0].message.content
    prediction = 1 if "PREDICTION: 1" in output else 0
    rationale = output.split("RATIONALE:")[-1].strip() if "RATIONALE:" in output else ""
    
    return prediction, rationale
```

### 7. Self-Consistency (Vote Ensemble)

```python
def classify_with_voting(backstory: str, evidence: list[str], num_votes: int = 5) -> tuple[int, str]:
    """Run multiple classifications and vote."""
    votes = []
    rationales = []
    
    for _ in range(num_votes):
        pred, rationale = classify_consistency(backstory, evidence)
        votes.append(pred)
        rationales.append(rationale)
    
    # Majority vote
    final_prediction = 1 if sum(votes) > num_votes // 2 else 0
    
    # Select rationale from the majority
    majority_rationales = [r for p, r in zip(votes, rationales) if p == final_prediction]
    final_rationale = majority_rationales[0] if majority_rationales else ""
    
    return final_prediction, final_rationale
```

---

## 📊 Evaluation Strategy

### Metrics to Track
- **Accuracy**: Overall correct predictions
- **Precision/Recall**: For each class (consistent/inconsistent)
- **F1 Score**: Balanced metric

### Testing Approach
1. Create a small validation set manually
2. Test with simple cases first
3. Analyze failure modes
4. Iterate on prompt engineering

---

## 📁 Recommended Project Structure

```
kdsh/
├── data/
│   ├── novels/           # Full novel texts
│   ├── backstories/      # Hypothetical backstories
│   └── test.json         # Test examples metadata
├── src/
│   ├── __init__.py
│   ├── ingest.py         # Pathway data ingestion
│   ├── chunker.py        # Chunking strategies
│   ├── embeddings.py     # Embedding generation
│   ├── vector_store.py   # Vector store operations
│   ├── retrieval.py      # Evidence retrieval
│   ├── reranker.py       # Reranking logic
│   ├── reasoning.py      # LLM-based classification
│   └── pipeline.py       # Main orchestration
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb
├── results.csv           # Output file
├── requirements.txt
├── README.md
└── report.pdf           # 10-page report
```

---

## ⚡ Quick Start Checklist

- [ ] Set up Python environment
- [ ] Install Pathway framework
- [ ] Download dataset from provided link
- [ ] Implement basic chunking
- [ ] Set up vector store
- [ ] Implement simple retrieval
- [ ] Create basic LLM prompts
- [ ] Run end-to-end pipeline
- [ ] Analyze results and iterate

---

## 📚 Learning Resources

### Pathway Framework
- [Pathway Documentation](https://pathway.com/docs)
- [Pathway LLM App Templates](https://github.com/pathwaycom/llm-app)
- [Pathway Bootcamp](https://pathway.com/bootcamp)

### RAG & Embeddings
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Docs](https://www.sbert.net/)

### Long Context Handling
- [LlamaIndex Long Context Guide](https://docs.llamaindex.ai/)
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)

### Prompt Engineering
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chain-of-Thought Prompting Paper](https://arxiv.org/abs/2201.11903)

---

## 🎯 Success Criteria (What Evaluators Want)

1. **Accuracy** - High classification performance
2. **Novel NLP Ideas** - Not just copy-paste RAG
3. **Long Context Handling** - Smart strategies for 100k+ words
4. **Evidence-Based** - Decisions backed by text excerpts
5. **Causal Reasoning** - Understanding cause-effect
6. **Reproducibility** - Clean, runnable code
7. **Clear Report** - Well-written 10-page document

Good luck with KDSH 2026! 🚀
