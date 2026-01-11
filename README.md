# KDSH Track A: Backstory Consistency Classification

A RAG-based pipeline for determining whether character backstories are consistent with long-form narratives.

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension (or use provided Neon database)
- Gemini API key

### 2. Install Dependencies

```bash
cd kdsh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Database (use your own or the provided Neon URL)
DATABASE_URL=postgresql://user:password@host/database?sslmode=require

# Optional
LLM_MODEL=gemini-2.5-flash
LLM_RPM_LIMIT=5
```

### 4. Prepare Data

Place the novel files and CSV data in the `Dataset/` folder:

```
Dataset/
├── Books/
│   ├── In search of the castaways.txt
│   └── The Count of Monte Cristo.txt
├── train.csv
└── test.csv
```

### 5. Generate Results

```bash
# Generate submission results (resumable if rate limited)
python3 run_pipeline.py --generate-results --input Dataset/test.csv --output results.csv
```

If you hit API rate limits, simply run the same command again - it will resume from the last saved row.

## Usage

### Generate Submission

```bash
python3 run_pipeline.py --generate-results --input Dataset/test.csv --output results.csv
```

### Evaluate on Training Data

```bash
python3 run_pipeline.py --evaluate --rows 10
```

### Index Books Only

```bash
python3 run_pipeline.py --index-only
```

## Pipeline Overview

Our system uses **Retrieval-Augmented Generation (RAG)** with **cross-encoder reranking** (70% accuracy).

```
Novel → Chunk (1000 words) → Embed → PostgreSQL/pgvector
                                           ↓
Backstory → Query → Vector Search (top-10) → Rerank (top-5) → LLM → Prediction
```

### Key Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Reranking | Enabled (default) | Best accuracy |
| Chunk size | 1000 words | Balance context/precision |
| Top-k retrieval | 10 → 5 after rerank | Focus on relevant evidence |

## Output Format

The `results.csv` file contains:

| Column | Description |
|--------|-------------|
| Story ID | Test example ID |
| Prediction | 1 = consistent, 0 = inconsistent |
| Rationale | Brief explanation |

## Troubleshooting

**Rate Limit Errors**: The pipeline handles rate limits with exponential backoff. If it fails after retries, wait a few minutes and run again - progress is saved.

**Database Connection**: Ensure `DATABASE_URL` is set correctly. The pipeline will create tables automatically.

**Missing Books**: Ensure novel `.txt` files are in `Dataset/Books/` with exact names matching the CSV.
