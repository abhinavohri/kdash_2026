"""Claim extraction for fine-grained backstory verification."""

import re
from typing import List
from dataclasses import dataclass

from ..logger import get_logger

logger = get_logger("claims")


@dataclass
class Claim:
    text: str
    claim_type: str
    entities: List[str]


class ClaimExtractor:
    """Extracts atomic claims from backstory text."""
    
    def __init__(self, use_llm: bool = False, llm_provider=None):
        self.use_llm = use_llm
        self.llm = llm_provider
        logger.info(f"ClaimExtractor initialized (use_llm={use_llm})")
    
    def extract_claims(self, backstory: str, character: str = "") -> List[Claim]:
        if self.use_llm and self.llm:
            return self._extract_with_llm(backstory, character)
        return self._extract_rule_based(backstory, character)
    
    def _extract_rule_based(self, backstory: str, character: str) -> List[Claim]:
        raw_claims = re.split(r'(?<=[.!?;])\s+', backstory)
        claims = []
        
        for raw in raw_claims:
            raw = raw.strip()
            if len(raw) < 15 or not raw:
                continue
            
            claims.append(Claim(
                text=raw,
                claim_type=self._classify_claim(raw),
                entities=self._extract_entities(raw, character)
            ))
        
        logger.info(f"Extracted {len(claims)} claims from backstory")
        return claims
    
    def _classify_claim(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["at age", "years old", "when", "before", "after", "during", "later"]):
            return "timeline"
        if any(w in text_lower for w in ["father", "mother", "brother", "sister", "friend", "enemy", "met", "married"]):
            return "relationship"
        if any(w in text_lower for w in ["always", "never", "believed", "feared", "loved", "hated", "was known"]):
            return "trait"
        return "event"
    
    def _extract_entities(self, text: str, primary_character: str) -> List[str]:
        entities = []
        if primary_character and primary_character.lower() in text.lower():
            entities.append(primary_character)
        
        words = text.split()
        for i, word in enumerate(words):
            if i == 0:
                continue
            if word[0].isupper() and word.lower() not in ["the", "a", "an", "he", "she", "his", "her", "it"]:
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word and clean_word not in entities:
                    entities.append(clean_word)
        return entities
    
    def _extract_with_llm(self, backstory: str, character: str) -> List[Claim]:
        prompt = f"""Extract atomic claims from this backstory.
Character: {character}
Backstory: {backstory}

Output format (one per line):
[EVENT] Claim text
[RELATIONSHIP] Claim text
[TRAIT] Claim text
[TIMELINE] Claim text"""
        
        try:
            response = self.llm.generate(prompt)
            return self._parse_llm_claims(response, character)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return self._extract_rule_based(backstory, character)
    
    def _parse_llm_claims(self, response: str, character: str) -> List[Claim]:
        claims = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[(\w+)\]\s*(.+)', line)
            if match:
                claims.append(Claim(
                    text=match.group(2),
                    claim_type=match.group(1).lower(),
                    entities=self._extract_entities(match.group(2), character)
                ))
        return claims


def get_claim_extractor(use_llm: bool = False, llm_provider=None) -> ClaimExtractor:
    return ClaimExtractor(use_llm=use_llm, llm_provider=llm_provider)
