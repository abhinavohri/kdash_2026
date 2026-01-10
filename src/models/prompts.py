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


PromptStrategy = Literal["base", "few_shot", "cot", "claim_extraction"]


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
# STRATEGY FACTORY
# =============================================================================

def get_prompt_template(strategy: PromptStrategy) -> str:
    """
    Get the prompt template for a given strategy.
    
    Args:
        strategy: One of 'base', 'few_shot', 'cot', 'claim_extraction'
        
    Returns:
        Prompt template string
    """
    templates = {
        "base": BASE_PROMPT,
        "few_shot": FEW_SHOT_PROMPT,
        "cot": COT_PROMPT,
        "claim_extraction": CLAIM_EXTRACTION_PROMPT,
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
