"""Memory and state tracking for multi-claim verification."""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..logger import get_logger

logger = get_logger("memory")


class ClaimVerdict(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


@dataclass
class ClaimEvidence:
    claim_text: str
    verdict: ClaimVerdict
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)


@dataclass
class EvidenceState:
    character: str
    book_name: str
    claims: Dict[str, ClaimEvidence] = field(default_factory=dict)
    supported_count: int = 0
    contradicted_count: int = 0
    neutral_count: int = 0
    
    def add_claim_result(self, claim_text: str, verdict: ClaimVerdict, confidence: float, evidence_chunk: str = ""):
        if claim_text not in self.claims:
            self.claims[claim_text] = ClaimEvidence(claim_text=claim_text, verdict=verdict, confidence=confidence)
        
        claim_ev = self.claims[claim_text]
        
        if verdict == ClaimVerdict.SUPPORTED:
            claim_ev.supporting_evidence.append(evidence_chunk[:200])
            if claim_ev.verdict == ClaimVerdict.NEUTRAL:
                claim_ev.verdict = ClaimVerdict.SUPPORTED
                self.supported_count += 1
                self.neutral_count -= 1
        elif verdict == ClaimVerdict.CONTRADICTED:
            claim_ev.contradicting_evidence.append(evidence_chunk[:200])
            if claim_ev.verdict != ClaimVerdict.CONTRADICTED:
                if claim_ev.verdict == ClaimVerdict.SUPPORTED:
                    self.supported_count -= 1
                elif claim_ev.verdict == ClaimVerdict.NEUTRAL:
                    self.neutral_count -= 1
                claim_ev.verdict = ClaimVerdict.CONTRADICTED
                self.contradicted_count += 1
        
        claim_ev.confidence = max(claim_ev.confidence, confidence)
    
    def get_strong_contradictions(self, min_confidence: float = 0.85) -> List[ClaimEvidence]:
        return [ce for ce in self.claims.values() if ce.verdict == ClaimVerdict.CONTRADICTED and ce.confidence >= min_confidence]
    
    def get_final_verdict(self, min_contradictions: int = 2, min_contradiction_confidence: float = 0.85) -> Tuple[int, str]:
        strong_contradictions = self.get_strong_contradictions(min_contradiction_confidence)
        
        if len(strong_contradictions) >= min_contradictions:
            examples = [c.claim_text[:50] + "..." for c in strong_contradictions[:2]]
            rationale = f"Found {len(strong_contradictions)} strong contradictions: {examples}"
            logger.info(f"Verdict: INCONSISTENT ({len(strong_contradictions)} contradictions)")
            return 0, rationale
        
        if self.supported_count > 0 and len(strong_contradictions) == 0:
            rationale = f"Found {self.supported_count} supporting evidence, no strong contradictions"
            logger.info(f"Verdict: CONSISTENT ({self.supported_count} supported)")
            return 1, rationale
        
        rationale = f"Insufficient evidence: {self.supported_count} supported, {len(strong_contradictions)} weak contradictions"
        logger.info(f"Verdict: CONSISTENT (default) - insufficient evidence")
        return 1, rationale


class MemoryAccumulator:
    """Accumulates evidence across multiple verification passes."""
    
    def __init__(self):
        self.states: Dict[str, EvidenceState] = {}
        logger.info("MemoryAccumulator initialized")
    
    def get_or_create_state(self, character: str, book_name: str) -> EvidenceState:
        key = f"{book_name}::{character}"
        if key not in self.states:
            self.states[key] = EvidenceState(character=character, book_name=book_name)
        return self.states[key]
    
    def record_nli_result(self, character: str, book_name: str, claim_text: str, 
                          nli_label: str, confidence: float, evidence_chunk: str = ""):
        state = self.get_or_create_state(character, book_name)
        verdict_map = {"entailment": ClaimVerdict.SUPPORTED, "contradiction": ClaimVerdict.CONTRADICTED, "neutral": ClaimVerdict.NEUTRAL}
        verdict = verdict_map.get(nli_label, ClaimVerdict.UNCERTAIN)
        state.add_claim_result(claim_text, verdict, confidence, evidence_chunk)
    
    def get_verdict(self, character: str, book_name: str, min_contradictions: int = 2, min_confidence: float = 0.85) -> Tuple[int, str]:
        state = self.get_or_create_state(character, book_name)
        return state.get_final_verdict(min_contradictions, min_confidence)
    
    def clear(self):
        self.states.clear()
        logger.info("Memory cleared")


def get_memory_accumulator() -> MemoryAccumulator:
    return MemoryAccumulator()
