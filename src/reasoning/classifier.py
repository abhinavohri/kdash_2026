"""
Consistency classifier using LLM reasoning.

Wraps LLM provider for backstory consistency classification.
"""

from typing import List, Tuple, Optional

from ..config import LLMConfig
from ..models.llm import get_llm_provider, LLMProvider
from ..logger import get_logger

logger = get_logger("classifier")


class ConsistencyClassifier:
    """
    Classifies whether character backstories are consistent with novel evidence.
    
    Uses LLM-based reasoning to analyze backstory claims against
    retrieved evidence from the novel.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: LLMConfig = None
    ):
        """
        Initialize the classifier.
        
        Args:
            llm_provider: Pre-configured LLM provider (optional)
            config: LLM configuration (if no provider given)
        """
        self.config = config or LLMConfig()
        
        if llm_provider:
            self.llm = llm_provider
        elif self.config.use_local:
            # Use Ollama for local inference
            from ..models.ollama_llm import get_ollama_llm
            self.llm = get_ollama_llm(model=self.config.local_model, config=self.config)
        else:
            # Use cloud API (Gemini)
            self.llm = get_llm_provider(config)
        
        model_name = self.config.local_model if self.config.use_local else self.config.model
        logger.info(f"Initialized ConsistencyClassifier with model={model_name} (local={self.config.use_local})")
    
    def classify(
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
        logger.info(f"Classifying backstory for '{character}' in '{book_name}'")
        
        if not evidence:
            logger.warning("No evidence provided, defaulting to consistent")
            return 1, "No evidence found to contradict the backstory"
        
        # Use the LLM to classify
        prediction, rationale = self.llm.classify_consistency(
            backstory=backstory,
            evidence=evidence,
            character=character,
            book_name=book_name
        )
        
        logger.info(f"Classification result: {'consistent' if prediction == 1 else 'inconsistent'}")
        
        return prediction, rationale
    
    def classify_batch(
        self,
        items: List[dict]
    ) -> List[Tuple[int, str]]:
        """
        Classify multiple backstories.
        
        Args:
            items: List of dictionaries with keys:
                   - backstory: str
                   - evidence: List[str]
                   - character: str
                   - book_name: str
            
        Returns:
            List of (prediction, rationale) tuples
        """
        results = []
        
        for i, item in enumerate(items):
            logger.info(f"Processing item {i + 1}/{len(items)}")
            
            pred, rationale = self.classify(
                backstory=item["backstory"],
                evidence=item["evidence"],
                character=item["character"],
                book_name=item["book_name"]
            )
            results.append((pred, rationale))
        
        return results


def get_classifier(config: LLMConfig = None) -> ConsistencyClassifier:
    """
    Factory function to create a consistency classifier.
    
    Args:
        config: LLM configuration
        
    Returns:
        Configured consistency classifier
    """
    return ConsistencyClassifier(config=config)
