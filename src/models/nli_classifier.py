"""
NLI (Natural Language Inference) based classifier.

Uses pre-trained NLI models to detect contradictions between
backstory and evidence. This is more direct than LLM prompting.
"""

from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import LLMConfig
from ..logger import get_logger

logger = get_logger("nli")


class NLIClassifier:
    """
    Classifier using Natural Language Inference.
    
    NLI models are trained on:
    - ENTAILMENT: Hypothesis follows from premise
    - CONTRADICTION: Hypothesis contradicts premise  
    - NEUTRAL: Neither
    
    For our task:
    - Premise = Evidence from novel
    - Hypothesis = Backstory claim
    
    If ANY evidence contradicts the backstory → INCONSISTENT
    """
    
    # Label indices for most NLI models
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base-tasksource-nli",
        device: str = None,
        contradiction_threshold: float = 0.7
    ):
        """
        Initialize NLI classifier.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on (auto-detected if None)
            contradiction_threshold: Min probability to flag as contradiction
        """
        self.model_name = model_name
        self.contradiction_threshold = contradiction_threshold
        
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading NLI model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"NLI model loaded successfully")
    
    def check_contradiction(
        self,
        premise: str,
        hypothesis: str
    ) -> Tuple[str, float]:
        """
        Check if hypothesis contradicts the premise.
        
        Args:
            premise: The evidence text
            hypothesis: The claim to verify
            
        Returns:
            Tuple of (label, confidence)
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get label and confidence
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        labels = ["entailment", "neutral", "contradiction"]
        label = labels[pred_idx]
        
        return label, confidence
    
    def classify_backstory(
        self,
        backstory: str,
        evidence: List[str],
        character: str = "",
        book_name: str = ""
    ) -> Tuple[int, str]:
        """
        Classify if backstory is consistent with evidence.
        
        Checks each evidence chunk for contradiction.
        If any chunk strongly contradicts → INCONSISTENT
        
        Args:
            backstory: Character backstory to verify
            evidence: List of relevant excerpts from novel
            character: Character name (for logging)
            book_name: Book name (for logging)
            
        Returns:
            Tuple of (prediction: 0 or 1, rationale)
        """
        logger.info(f"NLI classifying: {character} in {book_name}")
        
        if not evidence:
            return 1, "No evidence to check against"
        
        contradictions = []
        entailments = []
        
        for i, chunk in enumerate(evidence):
            label, confidence = self.check_contradiction(
                premise=chunk,
                hypothesis=backstory
            )
            
            logger.debug(f"Chunk {i+1}: {label} ({confidence:.2f})")
            
            if label == "contradiction" and confidence >= self.contradiction_threshold:
                contradictions.append((i, chunk[:100], confidence))
            elif label == "entailment" and confidence >= 0.7:
                entailments.append((i, chunk[:100], confidence))
        
        # Decision logic
        if contradictions:
            # Found contradiction → INCONSISTENT
            idx, snippet, conf = contradictions[0]
            rationale = f"Contradiction found in evidence {idx+1} (confidence: {conf:.2f}): '{snippet}...'"
            logger.info(f"Found {len(contradictions)} contradictions, marking INCONSISTENT")
            return 0, rationale
        elif entailments:
            # Found support → CONSISTENT
            idx, snippet, conf = entailments[0]
            rationale = f"Backstory supported by evidence {idx+1} (confidence: {conf:.2f})"
            logger.info(f"Found {len(entailments)} supporting evidence, marking CONSISTENT")
            return 1, rationale
        else:
            # Neutral → default to CONSISTENT (no contradiction found)
            logger.info("No strong signal, defaulting to CONSISTENT")
            return 1, "No clear contradiction or support found"
    
    def classify_consistency(
        self,
        backstory: str,
        evidence: List[str],
        character: str,
        book_name: str
    ) -> Tuple[int, str]:
        """Alias for classify_backstory to match LLM interface."""
        return self.classify_backstory(backstory, evidence, character, book_name)


def get_nli_classifier(
    model_name: str = "microsoft/deberta-v3-base-tasksource-nli",
    contradiction_threshold: float = 0.7
) -> NLIClassifier:
    """
    Factory function to get an NLI classifier.
    
    Args:
        model_name: HuggingFace model name
        contradiction_threshold: Min probability to flag as contradiction
        
    Returns:
        Configured NLI classifier
    """
    return NLIClassifier(
        model_name=model_name,
        contradiction_threshold=contradiction_threshold
    )
