"""
Canonical Backstory Generator.

Generates canonical backstories for characters from novel chunks.
"""

from typing import List, Optional
from ..config import PipelineConfig
from ..storage.pgvector import VectorStore
from ..models.embeddings import EmbeddingProvider
from ..models.llm import LLMProvider
from ..logger import get_logger

logger = get_logger("backstory_generator")


BACKSTORY_GENERATION_PROMPT = """You are a literary analyst. Your task is to summarize what the novel explicitly tells us about a character's backstory.

## Character: {character}
## Book: {book_name}

## Excerpts from the novel mentioning this character:
{evidence}

## Task:
Based ONLY on the excerpts above, write a concise summary of what the novel tells us about {character}'s:
- Origins (birth, family, early life)
- Key life events before the novel's main plot
- Relationships with other characters
- Background skills, knowledge, or traits

## Important:
- Only include information EXPLICITLY stated or strongly implied in the excerpts
- Do NOT invent or speculate about backstory not in the text
- If little backstory is provided, keep the summary brief
- Write in third person, past tense

## Output:
Write a 2-4 paragraph summary of {character}'s canonical backstory as established by the novel."""


class CanonicalBackstoryGenerator:
    """
    Generates canonical backstories for characters by analyzing novel chunks.
    
    Uses LLM to summarize character-relevant passages into a canonical backstory.
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider
    ):
        self.config = config
        self.store = vector_store
        self.embedder = embedding_provider
        self.llm = llm_provider
        
        # Ensure canonical schema exists
        self.store.init_canonical_schema()
        
        logger.info("Initialized CanonicalBackstoryGenerator")
    
    def generate_for_character(
        self,
        book_name: str,
        character: str,
        force_regenerate: bool = False
    ) -> str:
        """
        Generate canonical backstory for a character.
        
        1. Retrieve chunks mentioning the character
        2. Send to LLM to generate canonical backstory
        3. Store in database with embedding
        
        Args:
            book_name: Name of the book
            character: Character name
            force_regenerate: Regenerate even if exists
            
        Returns:
            Generated canonical backstory
        """
        logger.info(f"Generating canonical backstory for '{character}' in '{book_name}'")
        
        # Check if already exists
        if not force_regenerate:
            existing = self.store.get_canonical_backstory(book_name, character)
            if existing:
                logger.info(f"Canonical backstory already exists for '{character}'")
                return existing[0]
        
        # Retrieve chunks mentioning this character
        chunks = self.store.search_by_character(book_name, character, top_k=25)
        
        if not chunks:
            logger.warning(f"No chunks found mentioning '{character}' in '{book_name}'")
            return None
        
        # Format evidence
        evidence_text = "\n\n---\n\n".join([
            f"[Excerpt {i+1}]:\n{chunk}" 
            for i, chunk in enumerate(chunks)
        ])
        
        # Build prompt
        prompt = BACKSTORY_GENERATION_PROMPT.format(
            character=character,
            book_name=book_name,
            evidence=evidence_text
        )
        
        # Generate backstory via LLM
        backstory = self.llm.generate(prompt)
        
        # Generate embedding for the backstory
        embedding = self.embedder.embed_query(backstory)
        
        # Store in database
        self.store.store_canonical_backstory(
            book_name=book_name,
            character_name=character,
            backstory=backstory,
            embedding=embedding,
            model_used=self.config.llm.model
        )
        
        logger.info(f"Generated and stored canonical backstory for '{character}' ({len(backstory)} chars)")
        
        return backstory
    
    def generate_all(
        self,
        book_name: str,
        characters: List[str],
        force_regenerate: bool = False
    ) -> dict:
        """
        Generate canonical backstories for multiple characters.
        
        Args:
            book_name: Name of the book
            characters: List of character names
            force_regenerate: Regenerate even if exists
            
        Returns:
            Dict mapping character names to their backstories
        """
        logger.info(f"Generating backstories for {len(characters)} characters in '{book_name}'")
        
        results = {}
        for character in characters:
            backstory = self.generate_for_character(
                book_name, character, force_regenerate
            )
            results[character] = backstory
        
        logger.info(f"Generated {len(results)} canonical backstories")
        return results


def get_backstory_generator(
    config: PipelineConfig,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider
) -> CanonicalBackstoryGenerator:
    """Factory function for backstory generator."""
    return CanonicalBackstoryGenerator(
        config, vector_store, embedding_provider, llm_provider
    )
