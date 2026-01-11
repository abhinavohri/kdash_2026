# KDSH Track A: Backstory Consistency Classification

RAG pipeline for classifying whether character backstories are consistent with novels.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and DATABASE_URL
```

## Generate Results

```bash
python3 run_pipeline.py --generate-results --input Dataset/test.csv --output results.csv
```

This automatically indexes novels, generates character backstories, and classifies all test examples.

## Output

`results.csv` with columns: Story ID, Prediction (1=consistent, 0=inconsistent), Rationale
