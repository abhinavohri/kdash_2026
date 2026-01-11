"""Hybrid NLI + LLM classifier for backstory consistency."""

from typing import List, Tuple
from dataclasses import dataclass

from ..config import LLMConfig
from ..logger import get_logger
from .claims import Claim, get_claim_extractor
from .memory import get_memory_accumulator

logger = get_logger("hybrid_classifier")


@dataclass
class ClaimResult:
    claim: Claim
    verdict: str
    confidence: float
    evidence_snippet: str


class HybridClassifier:
    """Two-stage classifier: NLI for fast contradiction detection, LLM for uncertain cases."""
    
    def __init__(self, nli_classifier=None, llm_provider=None, config: LLMConfig = None,
                 nli_confidence_threshold: float = 0.9, min_contradictions: int = 2, use_llm_fallback: bool = True):
        self.config = config or LLMConfig()
        self.nli_threshold = nli_confidence_threshold
        self.min_contradictions = min_contradictions
        self.use_llm_fallback = use_llm_fallback
        self._nli = nli_classifier
        self._llm = llm_provider
        self._claim_extractor = get_claim_extractor(use_llm=False)
        self._memory = get_memory_accumulator()
        
        logger.info(f"HybridClassifier initialized: threshold={nli_confidence_threshold}, min_contradictions={min_contradictions}, llm_fallback={use_llm_fallback}")
    
    @property
    def nli(self):
        if self._nli is None:
            from ..models.nli_classifier import get_nli_classifier
            self._nli = get_nli_classifier(contradiction_threshold=self.nli_threshold)
        return self._nli
    
    @property
    def llm(self):
        if self._llm is None and self.use_llm_fallback:
            from ..models.llm import get_llm_provider
            self._llm = get_llm_provider(self.config)
        return self._llm
    
    def classify_consistency(self, backstory: str, evidence: List[str], character: str, book_name: str) -> Tuple[int, str]:
        logger.info(f"Hybrid classifying: {character} in {book_name}")
        
        if not evidence:
            return 1, "No evidence found to contradict the backstory"
        
        self._memory.clear()
        
        claims = self._claim_extractor.extract_claims(backstory, character)
        logger.info(f"Extracted {len(claims)} claims from backstory")
        
        if not claims:
            claims = [Claim(text=backstory, claim_type="event", entities=[character])]
        
        for claim in claims:
            for evidence_chunk in evidence:
                label, confidence = self.nli.check_contradiction(premise=evidence_chunk, hypothesis=claim.text)
                self._memory.record_nli_result(
                    character=character, book_name=book_name, claim_text=claim.text,
                    nli_label=label, confidence=confidence, evidence_chunk=evidence_chunk
                )
        
        state = self._memory.get_or_create_state(character, book_name)
        strong_contradictions = state.get_strong_contradictions(self.nli_threshold)
        
        if len(strong_contradictions) >= self.min_contradictions:
            examples = [c.claim_text[:60] for c in strong_contradictions[:2]]
            rationale = f"Found {len(strong_contradictions)} strong contradictions: {examples}"
            logger.info(f"NLI verdict: INCONSISTENT ({len(strong_contradictions)} contradictions)")
            return 0, rationale
        
        if state.supported_count > 0 and len(strong_contradictions) == 0:
            rationale = f"Backstory supported by {state.supported_count} evidence pieces, no contradictions"
            logger.info(f"NLI verdict: CONSISTENT ({state.supported_count} supported)")
            return 1, rationale
        
        if self.use_llm_fallback and self.llm:
            logger.info("NLI uncertain, falling back to LLM")
            return self._llm_classify(backstory, evidence, character, book_name, state)
        
        weak_contradictions = len(state.get_strong_contradictions(0.7))
        rationale = f"Insufficient evidence. Supported: {state.supported_count}, Weak contradictions: {weak_contradictions}"
        logger.info(f"NLI verdict: CONSISTENT (default)")
        return 1, rationale
    
    def _llm_classify(self, backstory: str, evidence: List[str], character: str, book_name: str, nli_state) -> Tuple[int, str]:
        nli_summary = f"Claims: {len(nli_state.claims)}, Supported: {nli_state.supported_count}, Contradicted: {nli_state.contradicted_count}"
        evidence_text = "\n\n---\n\n".join(evidence[:5])
        
        prompt = f"""Analyze if this backstory is CONSISTENT or INCONSISTENT with the novel.

Character: {character} | Book: {book_name}

Backstory: {backstory}

Evidence: {evidence_text}

NLI Analysis: {nli_summary}

Guidelines:
- CONSISTENT if it doesn't contradict the novel
- INCONSISTENT if it directly contradicts events/traits
- When ambiguous, favor CONSISTENT

Output:
PREDICTION: [1 for consistent, 0 for inconsistent]
RATIONALE: [One sentence]"""
        
        try:
            response = self.llm.generate(prompt)
            prediction, rationale = self._parse_llm_response(response)
            logger.info(f"LLM verdict: {'CONSISTENT' if prediction == 1 else 'INCONSISTENT'}")
            return prediction, f"(LLM) {rationale}"
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return nli_state.get_final_verdict(self.min_contradictions, self.nli_threshold)
    
    def _parse_llm_response(self, response: str) -> Tuple[int, str]:
        prediction = 1
        if "PREDICTION: 0" in response or "PREDICTION:0" in response:
            prediction = 0
        
        rationale = ""
        if "RATIONALE:" in response:
            rationale = response.split("RATIONALE:")[-1].strip()
        else:
            sentences = response.split(".")
            rationale = sentences[-2] + "." if len(sentences) > 1 else response[:200]
        
        return prediction, rationale


def get_hybrid_classifier(config: LLMConfig = None, nli_confidence_threshold: float = 0.9,
                          min_contradictions: int = 2, use_llm_fallback: bool = True) -> HybridClassifier:
    return HybridClassifier(config=config, nli_confidence_threshold=nli_confidence_threshold,
                            min_contradictions=min_contradictions, use_llm_fallback=use_llm_fallback)
