"""
Prompting strategies for classification.

Provides plug-and-play prompting approaches:
- base: Original chain-of-thought prompt
- few_shot: Add examples from training data
- cot: Enhanced chain-of-thought with explicit steps
- claim_extraction: Two-stage claim extraction then verification
"""

from typing import List, Literal
from dataclasses import dataclass


PromptStrategy = Literal["base", "few_shot", "cot", "claim_extraction", "track_a", "conservative", "optimistic", "evidence_dossier"]


# =============================================================================
# BASE PROMPT (Original)
# =============================================================================

BASE_PROMPT = """You are a literary analyst tasked with determining whether a character backstory is CONSISTENT or INCONSISTENT with a novel.

## Task
Analyze if the proposed backstory for {character} is consistent with the events and constraints in "{book_name}".

## Character Backstory to Verify:
{backstory}

## Relevant Excerpts from the Novel:
{evidence}

## Analysis Instructions:
1. Identify key claims in the backstory
2. Compare each claim against the evidence from the novel
3. Look for:
   - Direct contradictions (explicit conflicts with novel text)
   - Causal impossibilities (backstory events that would prevent novel events)
   - Character inconsistencies (backstory doesn't fit established character traits)
   - Timeline conflicts (events that couldn't happen in sequence)
4. Consider if the backstory could plausibly exist alongside the novel events

## Important:
- A backstory is CONSISTENT if it doesn't contradict the novel and could plausibly be true
- A backstory is INCONSISTENT if it directly contradicts events, character traits, or causal chains in the novel
- Minor ambiguities should favor CONSISTENT if no clear contradiction exists

## Output Format:
Provide your analysis, then conclude with exactly this format:

PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [One or two sentences explaining your decision]
"""


# =============================================================================
# FEW-SHOT PROMPT (With examples)
# =============================================================================

FEW_SHOT_PROMPT = """You are a literary analyst tasked with determining whether a character backstory is CONSISTENT or INCONSISTENT with a novel.

## Examples

### Example 1 (CONSISTENT):
**Character**: Thalcave
**Book**: In Search of the Castaways
**Backstory**: "Passing as a half-caste gaucho he worked on a ranch, picked up Spanish and the settlers' rules, yet guarded his tribe's herb lore and nature beliefs."
**Analysis**: This backstory aligns with Thalcave's characterization in the novel as a skilled Patagonian guide who knows both indigenous and settler ways. The novel shows him as knowledgeable about the land and respectful of traditions.
**PREDICTION**: 1
**RATIONALE**: The backstory of learning settler ways while preserving tribal knowledge is consistent with Thalcave's demonstrated dual cultural competence in the novel.

### Example 2 (INCONSISTENT):
**Character**: Tom Ayrton/Ben Joyce
**Book**: In Search of the Castaways
**Backstory**: "The mutiny began when Captain Grant uncovered his forged logbook and threatened to report him; 'marooning' was a botched silencing in which Ayrton, thinking fast, locked the captain inside the keel-less lifeboat."
**Analysis**: This directly contradicts the novel where Ayrton led the mutiny and deliberately marooned Captain Grant. The novel is clear that this was premeditated, not a "botched silencing."
**PREDICTION**: 0
**RATIONALE**: The backstory contradicts the novel's account of the mutiny being deliberate and planned, not accidental.

---

## Your Task
Analyze if the proposed backstory for {character} is consistent with the events and constraints in "{book_name}".

## Character Backstory to Verify:
{backstory}

## Relevant Excerpts from the Novel:
{evidence}

## Instructions:
1. Compare the backstory claims against the evidence
2. Look for direct contradictions, timeline conflicts, or character inconsistencies
3. A backstory is CONSISTENT if it could plausibly be true alongside the novel
4. A backstory is INCONSISTENT if it directly contradicts the novel

## Output Format:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [One or two sentences explaining your decision]
"""


# =============================================================================
# ENHANCED CHAIN-OF-THOUGHT PROMPT
# =============================================================================

COT_PROMPT = """You are a literary analyst. Your task is to determine if a character backstory is CONSISTENT or INCONSISTENT with a novel.

## Task
Analyze the backstory for {character} in "{book_name}".

## Character Backstory:
{backstory}

## Evidence from the Novel:
{evidence}

## Step-by-Step Analysis (You MUST complete each step):

### Step 1: Extract Claims
List the specific factual claims made in the backstory:
- Claim 1: [...]
- Claim 2: [...]
- Claim 3: [...]

### Step 2: Check Each Claim Against Evidence
For each claim, state whether the evidence supports, contradicts, or is silent:
- Claim 1: [SUPPORTS / CONTRADICTS / SILENT] because [...]
- Claim 2: [SUPPORTS / CONTRADICTS / SILENT] because [...]
- Claim 3: [SUPPORTS / CONTRADICTS / SILENT] because [...]

### Step 3: Look for Contradictions
Are there any:
- Direct contradictions with novel events? [YES/NO] - [explain]
- Timeline impossibilities? [YES/NO] - [explain]  
- Character trait conflicts? [YES/NO] - [explain]

### Step 4: Final Decision
Based on the above analysis:
- If ANY contradiction found → INCONSISTENT (0)
- If no contradictions and claims are plausible → CONSISTENT (1)

## Output:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [Summarize the key reason for your decision]
"""


# =============================================================================
# CLAIM EXTRACTION PROMPT (Two-stage)
# =============================================================================

CLAIM_EXTRACTION_PROMPT = """You are analyzing a character backstory to verify its consistency with a novel.

## Character: {character}
## Book: {book_name}

## Backstory to Analyze:
{backstory}

## Evidence from Novel:
{evidence}

## Task: Extract and Verify Claims

### Part 1: Extract all verifiable claims from the backstory
List each specific claim that could be checked against the novel:

1. [Claim about event/action]
2. [Claim about relationship]
3. [Claim about timeline]
... (list all claims)

### Part 2: Verdict for each claim
For each claim above, determine if the evidence:
- SUPPORTS it (evidence confirms this could be true)
- CONTRADICTS it (evidence shows this is false)
- NEUTRAL (no relevant evidence found)

Use format:
1. [SUPPORTS/CONTRADICTS/NEUTRAL]: [brief reason]
2. [SUPPORTS/CONTRADICTS/NEUTRAL]: [brief reason]
...

### Part 3: Final Classification
- If ANY claim is CONTRADICTED → The backstory is **INCONSISTENT**
- If all claims are SUPPORTED or NEUTRAL → The backstory is **CONSISTENT**

## Output:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [Which claim caused the contradiction, or why all claims are plausible]
"""


# =============================================================================
# EVIDENCE DOSSIER PROMPT (KDSH Problem Statement Requirements)
# =============================================================================

EVIDENCE_DOSSIER_PROMPT = """You are constructing an Evidence Dossier for the KDSH competition.

## CHARACTER: {character}
## BOOK: {book_name}

## BACKSTORY TO VERIFY:
{backstory}

## AVAILABLE EVIDENCE FROM NOVEL:
{evidence}

## YOUR TASK:
Construct a comprehensive Evidence Rationale following these STRICT requirements:

### STEP 1: CLAIM DECOMPOSITION
Split the backstory into ATOMIC claims (single verifiable facts).

### STEP 2: EVIDENCE LINKAGE
For EACH claim, search the evidence for VERBATIM passages that relate to it.

### STEP 3: ANALYSIS
For each claim-excerpt pair, provide analysis:
- SUPPORTS: Evidence confirms the claim
- CONTRADICTS: Evidence directly disproves the claim
- NEUTRAL: No relevant evidence found (absence is NOT contradiction)

### STEP 4: FINAL JUDGMENT
- If ANY claim has a CONTRADICTS verdict → PREDICTION: 0
- If ALL claims are SUPPORTS or NEUTRAL → PREDICTION: 1

## OUTPUT FORMAT (REQUIRED STRUCTURE):

EVIDENCE DOSSIER:

CLAIM 1: "[Verbatim atomic claim from backstory]"
EXCERPT: "[Verbatim quote from evidence, or 'No relevant passage found']"
VERDICT: [SUPPORTS/CONTRADICTS/NEUTRAL]
ANALYSIS: [How the excerpt constrains or refutes the claim]

CLAIM 2: "[Next atomic claim]"
EXCERPT: "[Verbatim quote from evidence]"
VERDICT: [SUPPORTS/CONTRADICTS/NEUTRAL]
ANALYSIS: [Explanation]

[Continue for all claims...]

---
FINAL JUDGMENT:
PREDICTION: [1 or 0]
SUMMARY: [One sentence explaining overall consistency based on evidence]
"""

# Keep old name for backwards compatibility
TRACK_A_PROMPT = EVIDENCE_DOSSIER_PROMPT


# =============================================================================
# CONSERVATIVE PROMPT (Default to Consistent)
# =============================================================================

CONSERVATIVE_PROMPT = """You are analyzing whether a character backstory is consistent with a novel.

## CRITICAL INSTRUCTION:
A backstory should be marked CONSISTENT (1) UNLESS you find EXPLICIT, DIRECT contradictions.
- "Not mentioned" = CONSISTENT (absence of evidence is not evidence of contradiction)
- "Could be different" = CONSISTENT (speculation is not contradiction)  
- "Seems unlikely" = CONSISTENT (improbability is not impossibility)
- ONLY mark INCONSISTENT (0) if the backstory DIRECTLY CONTRADICTS a specific fact in the novel

## Character: {character}
## Book: {book_name}

## Backstory:
{backstory}

## Evidence from Novel:
{evidence}

## Decision Process:
1. Read the backstory claims
2. Check if ANY claim DIRECTLY CONTRADICTS a specific statement in the evidence
3. If you cannot point to an EXPLICIT contradiction → answer CONSISTENT (1)
4. Only answer INCONSISTENT (0) if you can quote the exact contradiction

## Output (KEEP RATIONALE TO 1-2 LINES MAX):
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [1-2 sentence summary. If inconsistent, state the contradiction. If consistent, state "No contradictions found."]
"""


# =============================================================================
# OPTIMISTIC PROMPT (Default consistent unless proof of contradiction)
# =============================================================================

OPTIMISTIC_PROMPT = """You are analyzing whether a character backstory is consistent with a novel.

## CRITICAL INSTRUCTION - OPTIMISTIC MODE:
You MUST default to CONSISTENT (1) unless you find EXPLICIT, UNDENIABLE proof of contradiction.

The backstory is CONSISTENT (1) unless:
- There is a DIRECT, EXPLICIT statement in the evidence that PROVES the backstory claim is false
- The contradiction is UNAMBIGUOUS and CERTAIN (not speculative)

Mark as CONSISTENT (1) if:
- The claim is "not mentioned" in the evidence
- The claim "could be different" but isn't proven false
- The claim "seems unlikely" but isn't impossible
- You're uncertain or the evidence is ambiguous
- The claim adds details not covered in the novel

ONLY mark as INCONSISTENT (0) if:
- You can quote an EXACT passage that DIRECTLY PROVES the backstory is false
- The contradiction is 100% certain, not just probable

## Character: {character}
## Book: {book_name}

## Backstory:
{backstory}

## Evidence from Novel:
{evidence}

## Decision Process:
1. Read the backstory claims
2. Search for PROOF that any claim is false (not just absence of confirmation)
3. If you cannot find EXPLICIT PROOF of contradiction → CONSISTENT (1)
4. Only answer INCONSISTENT (0) if you have undeniable proof

## Output:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [If inconsistent, you MUST quote the exact proof of contradiction. If consistent, state why no proof of contradiction was found.]
"""


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

def get_prompt_template(strategy: PromptStrategy) -> str:
    templates = {
        "base": BASE_PROMPT,
        "few_shot": FEW_SHOT_PROMPT,
        "cot": COT_PROMPT,
        "claim_extraction": CLAIM_EXTRACTION_PROMPT,
        "track_a": TRACK_A_PROMPT,
        "conservative": CONSERVATIVE_PROMPT,
        "optimistic": OPTIMISTIC_PROMPT,
        "evidence_dossier": EVIDENCE_DOSSIER_PROMPT,
    }
    
    if strategy not in templates:
        raise ValueError(f"Unknown prompt strategy: {strategy}. Choose from: {list(templates.keys())}")
    
    return templates[strategy]


def format_prompt(
    strategy: PromptStrategy,
    character: str,
    book_name: str,
    backstory: str,
    evidence: str
) -> str:
    """
    Format a prompt with the given strategy and inputs.
    
    Args:
        strategy: Prompting strategy to use
        character: Character name
        book_name: Book name
        backstory: Backstory text
        evidence: Formatted evidence text
        
    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(strategy)
    
    return template.format(
        character=character,
        book_name=book_name,
        backstory=backstory,
        evidence=evidence
    )
