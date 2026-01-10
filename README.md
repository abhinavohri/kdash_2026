# KDSH Track A Pipeline

A modular, plug-and-play pipeline for the Kharagpur Data Science Hackathon 2026.

## 🎯 Task

Given a novel (100k+ words) and a hypothetical character backstory, classify whether the backstory is **consistent (1)** or **inconsistent (0)** with the novel.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for PostgreSQL + pgvector)
- Gemini API key

### 2. Setup

```bash
# Clone/navigate to project
cd kdsh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Start Database

```bash
# Start PostgreSQL with pgvector
docker-compose up -d

# Verify it's running
docker ps
# Should show: kdsh_postgres container running on port 5432
```

### 4. Run Evaluation

```bash
# Evaluate on 10 training examples
python run_pipeline.py --evaluate --rows 10

# Evaluate on more rows for reliable results
python run_pipeline.py --evaluate --rows 50
```

### 5. Generate Submission

```bash
# Generate results.csv for test data
python run_pipeline.py --generate-results
```

---

## 📁 Project Structure

```
kdsh/
├── src/
│   ├── config.py          # Central configuration (plug-and-play)
│   ├── logger.py          # Colored logging
│   ├── pipeline.py        # Main orchestration
│   ├── models/
│   │   ├── base.py        # Abstract interfaces
│   │   ├── embeddings.py  # Gemini embeddings
│   │   └── llm.py         # Gemini LLM
│   ├── data/
│   │   ├── chunking.py    # Chunking strategies
│   │   └── ingestion.py   # Pathway data loading
│   ├── storage/
│   │   └── pgvector.py    # PostgreSQL + pgvector
│   ├── retrieval/
│   │   └── retriever.py   # Evidence retrieval
│   └── reasoning/
│       └── classifier.py  # LLM classification
├── Dataset/
│   ├── Books/             # Novel text files
│   ├── train.csv          # Training data with labels
│   └── test.csv           # Test data for submission
├── docker-compose.yml     # PostgreSQL + pgvector
├── requirements.txt       # Python dependencies
├── run_pipeline.py        # Main entry point
├── evaluate.py            # Quick evaluation script
└── .env                   # API keys (create from .env.example)
```

---

## ⚙️ Configuration

All configuration is in `src/config.py`. To change parameters:

### Change LLM Model

```python
# In code
config.llm.model = "gemini-2.5-pro"

# Or via CLI
python run_pipeline.py --evaluate --rows 10 --model gemini-2.5-pro
```

### Change Embedding Model

```python
config.embedding.model = "gemini-embedding-001"
config.embedding.dimensions = 768  # or 1536, 3072
```

### Change Chunking Strategy

```python
config.chunking.strategy = "fixed_overlap"  # or "sentence"
config.chunking.chunk_size = 1000
config.chunking.overlap = 200
```

### Change Retrieval Parameters

```python
config.retrieval.top_k = 15  # Retrieve more chunks
```

---

## 🔌 Plug-and-Play Architecture

### Adding a New Embedding Provider

1. Create class in `src/models/embeddings.py`:
```python
class OpenAIEmbedding(EmbeddingProvider):
    def embed_query(self, text: str) -> List[float]:
        # Implementation
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Implementation
```

2. Update factory function:
```python
def get_embedding_provider(config):
    if config.provider == "openai":
        return OpenAIEmbedding(config)
```

3. Use it:
```python
config.embedding.provider = "openai"
```

### Adding a New Chunking Strategy

1. Create class in `src/data/chunking.py`:
```python
class SemanticChunker(ChunkingStrategy):
    def chunk(self, text: str) -> List[str]:
        # Implementation
```

2. Update factory in same file.

---

## 📊 Commands Reference

| Command | Description |
|---------|-------------|
| `python run_pipeline.py --evaluate --rows 10` | Evaluate on 10 training examples |
| `python run_pipeline.py --evaluate --rows 50` | Evaluate on 50 examples |
| `python run_pipeline.py --generate-results` | Generate submission CSV |
| `python run_pipeline.py --index-only` | Only index books |
| `python run_pipeline.py --reindex` | Force reindex all books |
| `python evaluate.py --rows 10` | Quick evaluation |

### CLI Options

```
--rows N          Number of rows for evaluation (default: 10)
--model MODEL     LLM model (e.g., gemini-2.5-flash, gemini-2.5-pro)
--top-k N         Chunks to retrieve (default: 10)
--chunk-size N    Words per chunk (default: 1000)
--reindex         Force reindex books
--output FILE     Output file path (default: results.csv)
```

---

## 🔧 Troubleshooting

### Database Connection Error

```bash
# Ensure Docker is running
docker-compose up -d

# Check container logs
docker logs kdsh_postgres
```

### API Key Error

```bash
# Check .env file exists and has key
cat .env | grep GEMINI_API_KEY
```

### Module Not Found

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📝 Output Format

### results.csv (Submission)

```csv
Story ID,Prediction,Rationale
1,1,Earlier economic shock makes outcome necessary
2,0,Proposed backstory contradicts later actions
```

### Evaluation Output

```
============================================================
Evaluation Results
============================================================
Accuracy: 70.00% (7/10)
  consistent: 80.00% (4/5)
  inconsistent: 60.00% (3/5)
```

---

## 🏆 Track A Requirements

✅ Uses **Pathway** framework for data ingestion  
✅ Uses **Gemini** API for embeddings and LLM  
✅ Uses **pgvector** for vector storage  
✅ Reproducible end-to-end pipeline  
✅ Generates required output format
