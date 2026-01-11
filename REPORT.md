# KDSH Track A: Technical Report

## 1. Overall Approach

We built a **Retrieval-Augmented Generation (RAG) pipeline** that retrieves relevant passages from novels and uses an LLM to determine whether character backstories are consistent with the source text.

### Our Journey

We started with a basic approach: chunk the novel into segments, embed them with Google's Gemini embedding model, store them in PostgreSQL with pgvector, retrieve the most similar chunks for each backstory query, and have Gemini classify whether the backstory is consistent.

Our initial accuracy was around 50%, which was essentially random chance. We quickly realized that the problem wasn't the LLM's reasoning ability—it was that we were feeding it the wrong evidence. The retrieved chunks were topically similar but not truly relevant to the specific claims in the backstory.

### What We Tried and What Worked

We ran over 15 experiments systematically varying different components:

| Experiment | Accuracy | What We Learned |
|------------|----------|-----------------|
| Baseline (top-k=20) | 50.0% | More chunks didn't help—we were just adding noise |
| Reranking with cross-encoder | **70.0%** | This was our biggest breakthrough |
| Chain-of-thought prompting | 60.0% | Reasoning helped slightly but not as much as better retrieval |
| Few-shot examples | 50.0% | Examples from other characters didn't generalize |
| Claim extraction (two-stage) | 40.0% | Breaking backstories into claims actually hurt—too fragmented |
| Pure NLI classification | 26.5% | Way too aggressive at finding "contradictions" |
| Hybrid NLI + LLM | 45.0% | NLI was over-flagging, dragging down accuracy |
| Conservative prompting | 62.5% | Defaulting to consistent helped with false positives |
| Canonical backstory approach | **90.0%** | Pre-generating character summaries worked well |

The key insight we discovered: **retrieval quality matters far more than retrieval quantity**. When we added cross-encoder reranking, our accuracy jumped from 50% to 70% overnight. The reranker (ms-marco-MiniLM-L-6-v2) scores each (backstory, chunk) pair jointly, which captures relevance much better than simple embedding similarity.

### Our Final Architecture

Our production system works as follows:

1. **Ingestion**: We read the novel text files and chunk them into 1000-word segments with 200-word overlap between consecutive chunks.

2. **Embedding**: Each chunk is embedded using Gemini's embedding-001 model (768 dimensions) and stored in PostgreSQL with the pgvector extension.

3. **Retrieval**: For each backstory query, we retrieve the top 20 chunks by cosine similarity with the backstory embedding.

4. **Reranking**: We re-score those 20 chunks using a cross-encoder and keep only the top 5 most relevant ones. This step is crucial—it filters out the "semantically similar but not actually relevant" chunks.

5. **Classification**: We send the backstory and the 5 reranked evidence chunks to Gemini 2.5 Flash with a conservative prompt that defaults to "consistent" unless there's explicit proof of contradiction.

6. **Rationale Generation**: The LLM outputs both a prediction (1 or 0) and a short rationale explaining its decision.

---

## 2. How We Handle Long Context (100k+ Words)

The novels in this dataset are massive—"The Count of Monte Cristo" is over 464,000 words. We can't fit an entire novel in any LLM's context window, so we needed a chunking and retrieval strategy.

### Our Chunking Decisions

We experimented with three chunk sizes:

- **500 words**: Too small. Important context got split across chunks. For example, if a character's backstory spans two paragraphs, we'd often get only half the relevant information.

- **2000 words**: Too large. Chunks became imprecise—they contained too much irrelevant text, diluting the signal. The embedding became a "summary of everything" rather than capturing specific details.

- **1000 words with 200-word overlap**: This was our sweet spot. Chunks were large enough to preserve context but small enough to be specific. The overlap ensures that information at chunk boundaries isn't lost.

### Why Overlap Matters

We initially tried chunks without overlap and found that critical passages often fell at the boundary between chunks. For instance, a sentence like "Thalcave, whose father had been the last of the pampas guides, knew every trail" might get split, with "Thalcave" in one chunk and "pampas guides" in another. The 200-word overlap ensures both chunks contain the complete context.

### Why We Limited Retrieved Chunks

Counterintuitively, retrieving more chunks hurt our accuracy. We tried retrieving 20 chunks and sending all of them to the LLM, but this just added noise. The LLM would find spurious "contradictions" in irrelevant passages.

Our solution was to retrieve broadly (20 chunks) but filter aggressively (keep 5 after reranking). The cross-encoder is crucial here—it can recognize when a chunk is topically related but doesn't actually provide evidence for or against the backstory.

---

## 3. How We Distinguish Causal Signals from Noise

This was our hardest challenge. The LLM was good at finding contradictions—too good. It would hallucinate contradictions that didn't exist or flag things as "inconsistent" when the evidence was merely silent.

### The Problem We Observed

Looking at our error analysis:

- **Inconsistent cases (actual=0)**: We got 100% accuracy. When a backstory truly contradicted the novel, the model reliably caught it.

- **Consistent cases (actual=1)**: We struggled badly, starting at 0% accuracy and eventually reaching 62.5% with our best approach.

The pattern was clear: our model had a strong bias toward predicting "inconsistent." It was finding contradictions where none existed.

### Example of a False Positive

One backstory said a character "loved geography." The model found a passage about the character studying astronomy and flagged this as a "contradiction"—arguing that loving geography contradicts studying astronomy. But obviously, someone can love geography AND study astronomy. This is exactly the kind of spurious reasoning we needed to eliminate.

### What Helped

**1. Cross-encoder reranking.** The reranker learned to distinguish relevant evidence from superficial matches. A chunk containing the word "geography" isn't necessarily relevant if the context is about a different character. The reranker understands this; pure embedding similarity doesn't.

**2. Conservative prompting.** We engineered our prompts to be skeptical of "contradictions." We explicitly told the model:
- "Not mentioned" does NOT mean contradiction
- "Seems unlikely" does NOT mean contradiction
- Only flag as inconsistent if there's EXPLICIT, DIRECT proof

This reduced false positives significantly.

**3. Optimistic mode.** For particularly stubborn cases, we added an "optimistic" flag that makes the model even more conservative—defaulting to consistent unless the contradiction is "undeniable and certain."

### What Didn't Work

**Claim extraction.** We tried a two-stage approach: first extract individual claims from the backstory ("Claim 1: Character was born in Paris", "Claim 2: Grew up orphaned"), then verify each claim separately. This backfired badly. By fragmenting the backstory, we lost important context. A claim like "grew up orphaned" might be consistent with the novel even if the novel doesn't explicitly mention it—but when analyzed in isolation, the model would flag it as "unverified" and sometimes call it inconsistent.

**NLI models.** We tried using Natural Language Inference models (specifically DeBERTa) to detect contradictions between the backstory and evidence chunks. The NLI model was extremely trigger-happy, calling almost everything a "contradiction." Even with a high confidence threshold (0.9), it was too aggressive. We suspect this is because NLI models are trained on short sentence pairs, not long passages.

**Few-shot examples.** We provided 2-3 examples of consistent and inconsistent backstories in the prompt. Surprisingly, this didn't help at all. We believe the examples were too specific to their particular characters and didn't generalize.

---

## 4. Advanced Approach: Canonical Backstories

Late in our development, we implemented an alternative approach that showed promising results: **pre-generating canonical backstories** for each character.

### The Concept

Instead of comparing each backstory against raw novel chunks, we first generated a "canonical backstory" for each character by:

1. Finding all chunks that mention the character (using simple string matching)
2. Sending those chunks to the LLM with a prompt asking it to summarize what the novel explicitly tells us about the character's background
3. Storing this canonical backstory in our database

Then, at classification time, we compare the input backstory against this canonical summary, asking: "Does this input contradict the canonical backstory?"

### Results

With the LLM-based canonical comparison, we achieved **90% accuracy** on our 10-row evaluation set. This is a significant improvement over the 70% from our standard RAG approach.

The canonical approach has several advantages:
- The canonical backstory is cleaner and more focused than raw chunks
- The comparison is more direct—we're comparing two backstories rather than a backstory and a pile of text
- It requires only 6 LLM calls to generate the canonicals (one per character), then comparison is straightforward

### Limitations

The canonical approach relies on correctly extracting all relevant information about a character. If the character is referred to by aliases or pronouns that we don't catch, we might miss important context. For example, if the novel says "Mary's brother was a sailor" without explicitly naming "John," our search for "John" would miss this passage.

---

## 5. Key Limitations and Failure Cases

### Limitation 1: Bias Toward Inconsistent

Despite our efforts, all our methods still struggle with consistent cases. The best we achieved was 62.5% accuracy on consistent cases (vs 100% on inconsistent). We believe this is an inherent asymmetry:

- Finding a contradiction requires only one piece of evidence
- Proving consistency requires checking against the entire novel (impossible)

We mitigated this with conservative prompting, but it remains our biggest weakness.

### Limitation 2: No Global Coherence

Our pipeline reasons over individual chunks, not the full narrative arc. A backstory might be globally inconsistent—contradicting character development that spans multiple chapters—while being locally plausible in any single chunk.

We started implementing hierarchical summarization to address this, but didn't have time to complete it.

### Limitation 3: Rate Limits and Evaluation Sample Size

We hit Gemini's free tier limit (5 RPM) frequently during experiments. Most of our experiments used only 10 rows for iteration speed. Our reported 70% accuracy is based on this small sample and might not generalize perfectly.

We made our pipeline resumable to handle rate limits gracefully—it saves progress after each row and can resume from any interruption.

### Limitation 4: Character Aliases and Pronouns

Our character-based retrieval uses simple string matching. If the novel refers to a character by nickname, title, or pronoun, we might miss relevant passages. This is particularly problematic for characters like "Tom Ayrton/Ben Joyce" who go by multiple names.

---

## 6. Technical Implementation Details

### Infrastructure

- **Database**: PostgreSQL with pgvector extension (hosted on Neon)
- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: Gemini embedding-001 (768 dimensions)
- **Reranker**: ms-marco-MiniLM-L-6-v2 cross-encoder
- **Local alternative**: Ollama with llama3.2 (lower accuracy but no API costs)

### Code Structure

Our codebase is organized into:
- `src/data/`: Ingestion and chunking logic
- `src/models/`: LLM and embedding providers, prompt templates
- `src/storage/`: PostgreSQL/pgvector interface
- `src/retrieval/`: Evidence retrieval and reranking
- `src/reasoning/`: Classification logic, canonical backstory generator

### Prompt Engineering

We developed multiple prompt strategies:
- `conservative`: Defaults to consistent, requires explicit proof of contradiction
- `optimistic`: Even more conservative, treats uncertainty as consistency
- `evidence_dossier`: Produces structured output with claim-excerpt-verdict triples
- `track_a`: Original structured format with explicit linkage

The conservative prompt is our default for submissions.

---

## 7. Conclusion

Our best approach is **RAG with cross-encoder reranking** using **conservative prompting**:

- **70% overall accuracy** (vs 50% baseline)
- **62.5% on consistent cases** (vs 0% baseline)  
- **100% on inconsistent cases** (maintained)

The canonical backstory approach shows even better results (90%) but requires additional preprocessing.

Our key insight: **retrieval quality matters more than retrieval quantity**. A smaller set of highly relevant chunks, selected by a cross-encoder, dramatically outperforms a larger set of loosely relevant ones.

The remaining challenge is reducing false positives on consistent cases. We believe this requires either:
1. Better global understanding of the novel (beyond chunk-level retrieval)
2. More sophisticated reasoning about what constitutes "consistency" vs "not mentioned"
3. Training data to fine-tune models for this specific task

We're confident our approach is sound and our results are reproducible. The pipeline handles rate limits gracefully and can generate predictions for any test set automatically.
