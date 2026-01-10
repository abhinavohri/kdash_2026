"""
LLM providers implementation.

Supports Gemini models for reasoning and classification.
"""

import os
import re
import time
from typing import List, Tuple
import google.generativeai as genai

from .base import LLMProvider
from ..config import LLMConfig
from ..logger import get_logger

logger = get_logger("llm")


CLASSIFICATION_PROMPT = """You are a literary analyst tasked with determining whether a character backstory is CONSISTENT or INCONSISTENT with a novel.

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


class GeminiLLM(LLMProvider):
    """
    Gemini LLM provider for reasoning and classification.
    
    Supports gemini-2.5-flash, gemini-2.5-pro, and other Gemini models.
    Includes rate limiting for free tier (5 RPM).
    """
    
    def __init__(self, config: LLMConfig = None):
        """
        Initialize Gemini LLM provider.
        
        Args:
            config: LLM configuration (uses defaults if None)
        """
        self.config = config or LLMConfig()
        self._last_call_time = 0
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.config.model)
        
        # Calculate delay based on RPM
        self._delay = 60.0 / self.config.rpm_limit if self.config.rpm_limit > 0 else 0
        
        logger.info(f"Initialized Gemini LLM: model={self.config.model}, rpm={self.config.rpm_limit}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response from Gemini.
        
        Includes rate limiting and automatic retry for quota errors.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        from google.api_core.exceptions import ResourceExhausted
        
        # Rate limiting
        if self._delay > 0:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._delay:
                wait_time = self._delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self._last_call_time = time.time()
        
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        # Retry with exponential backoff
        max_retries = 5
        base_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_tokens,
                    )
                )
                return response.text
                
            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    # Extract retry delay from error if available
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit: max retries exceeded")
                    raise
    
    def classify_consistency(
        self, 
        backstory: str, 
        evidence: List[str],
        character: str,
        book_name: str
    ) -> Tuple[int, str]:
        """
        Classify whether a backstory is consistent with the novel.
        
        Uses configurable prompting strategy for better reasoning.
        
        Args:
            backstory: Character backstory to verify
            evidence: List of relevant excerpts from the novel
            character: Character name
            book_name: Name of the book
            
        Returns:
            Tuple of (prediction: 0 or 1, rationale: explanation)
        """
        from .prompts import format_prompt
        
        logger.info(f"Classifying consistency for {character} in {book_name}")
        logger.debug(f"Using prompt strategy: {self.config.prompt_strategy}")
        
        # Format evidence
        evidence_text = "\n\n---\n\n".join([
            f"[Excerpt {i+1}]:\n{excerpt}" 
            for i, excerpt in enumerate(evidence)
        ])
        
        # Build prompt using selected strategy
        prompt = format_prompt(
            strategy=self.config.prompt_strategy,
            character=character,
            book_name=book_name,
            backstory=backstory,
            evidence=evidence_text
        )
        
        # Generate response
        response = self.generate(prompt)
        
        # Parse response
        prediction, rationale = self._parse_response(response)
        
        logger.info(f"Classification result: prediction={prediction}")
        logger.debug(f"Rationale: {rationale}")
        
        return prediction, rationale
    
    def _parse_response(self, response: str) -> Tuple[int, str]:
        """
        Parse the LLM response to extract prediction and rationale.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Tuple of (prediction, rationale)
        """
        # Default values
        prediction = 0
        rationale = ""
        
        # Extract prediction
        prediction_match = re.search(r"PREDICTION:\s*(\d)", response, re.IGNORECASE)
        if prediction_match:
            prediction = int(prediction_match.group(1))
        else:
            # Fallback: look for keywords
            lower_response = response.lower()
            if "consistent" in lower_response and "inconsistent" not in lower_response:
                prediction = 1
            elif "inconsistent" in lower_response or "contradict" in lower_response:
                prediction = 0
            logger.warning("Could not parse PREDICTION from response, using fallback")
        
        # Extract rationale
        rationale_match = re.search(r"RATIONALE:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        else:
            # Use last few sentences as fallback
            sentences = response.split(".")
            rationale = ". ".join(sentences[-3:]).strip() if len(sentences) >= 3 else response[-200:]
        
        return prediction, rationale


def get_llm_provider(config: LLMConfig = None) -> LLMProvider:
    """
    Factory function to get an LLM provider.
    
    Args:
        config: LLM configuration
        
    Returns:
        Configured LLM provider
    
    Raises:
        ValueError: If provider is not supported
    """
    config = config or LLMConfig()
    
    if config.provider == "gemini":
        return GeminiLLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
