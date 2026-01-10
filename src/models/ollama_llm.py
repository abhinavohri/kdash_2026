"""
Ollama LLM provider for local model inference.

Supports running local models without API rate limits.
"""

import time
import requests
from typing import List, Tuple, Optional

from ..config import LLMConfig
from ..logger import get_logger

logger = get_logger("ollama")

OLLAMA_API_BASE = "http://localhost:11434"


class OllamaLLM:
    """
    LLM provider using Ollama for local inference.
    
    No rate limits - runs entirely on your machine!
    """
    
    def __init__(self, config: LLMConfig = None, model: str = None):
        """
        Initialize Ollama LLM.
        
        Args:
            config: LLM configuration
            model: Model name (overrides config)
        """
        self.config = config or LLMConfig()
        self.model = model or self.config.model
        
        # Check if Ollama is running
        self._check_ollama()
        
        logger.info(f"Initialized Ollama LLM: model={self.model}")
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama is not responding")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if self.model.split(":")[0] not in model_names:
                available = ", ".join(model_names) if model_names else "none"
                logger.warning(f"Model '{self.model}' not found. Available: {available}")
                logger.info(f"Pull it with: ollama pull {self.model}")
            else:
                logger.info(f"Model '{self.model}' is available")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama is not running. Start it with: ollama serve\n"
                "Or install from: https://ollama.ai"
            )
    
    def generate(self, prompt: str) -> str:
        """
        Generate a response using Ollama.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            },
            timeout=300  # 5 min timeout for long responses
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.text}")
        
        result = response.json()
        return result.get("response", "")
    
    def classify_consistency(
        self, 
        backstory: str, 
        evidence: List[str],
        character: str,
        book_name: str
    ) -> Tuple[int, str]:
        """
        Classify whether a backstory is consistent with the novel.
        
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
        """Parse LLM response to extract prediction and rationale."""
        import re
        
        # Default values
        prediction = 1  # Default to consistent
        rationale = "No clear rationale provided"
        
        # Try to extract PREDICTION
        pred_match = re.search(r'PREDICTION:\s*(\d)', response, re.IGNORECASE)
        if pred_match:
            prediction = int(pred_match.group(1))
        else:
            # Fallback: look for keywords
            response_lower = response.lower()
            if 'inconsistent' in response_lower or 'contradict' in response_lower:
                prediction = 0
            elif 'consistent' in response_lower:
                prediction = 1
            logger.warning("Could not parse PREDICTION from response, using fallback")
        
        # Try to extract RATIONALE
        rat_match = re.search(r'RATIONALE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if rat_match:
            rationale = rat_match.group(1).strip()
        else:
            # Use last sentence as rationale
            sentences = response.split('.')
            if sentences:
                rationale = sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip()
        
        return prediction, rationale


def get_ollama_llm(model: str = "llama3.1:8b", config: LLMConfig = None) -> OllamaLLM:
    """
    Factory function to get an Ollama LLM.
    
    Args:
        model: Ollama model name
        config: LLM configuration
        
    Returns:
        Configured Ollama LLM
    """
    return OllamaLLM(config=config, model=model)
