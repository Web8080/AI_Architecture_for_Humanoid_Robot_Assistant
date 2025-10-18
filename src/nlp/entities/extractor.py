"""
Entity Extractor with 3-Tier Fallback System
Tier 1: BERT-based NER (Hugging Face)
Tier 2: Custom domain-specific NER (fine-tuned)
Tier 3: spaCy transformer NER

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from enum import Enum

# Tier 1: Hugging Face Transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")

# Tier 3: spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Install with: pip install spacy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Supported entity types for robotics domain"""
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    OBJECT = "OBJECT"
    COMMAND = "COMMAND"
    TIME = "TIME"
    DATE = "DATE"
    NUMBER = "NUMBER"
    COLOR = "COLOR"
    DIRECTION = "DIRECTION"
    ROOM = "ROOM"
    SENSOR = "SENSOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    type: str
    start: int
    end: int
    confidence: float
    tier: str  # Which tier extracted this entity


class EntityExtractor:
    """
    Multi-tier entity extraction with automatic fallback.
    Automatically detects GPU/CPU and adjusts accordingly.
    """
    
    def __init__(
        self,
        bert_model_name: str = "dslim/bert-base-NER",
        custom_model_path: Optional[str] = None,
        spacy_model: str = "en_core_web_trf",
        use_gpu: Optional[bool] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize entity extractor with multi-tier fallback.
        
        Args:
            bert_model_name: Hugging Face model name for Tier 1
            custom_model_path: Path to custom fine-tuned model for Tier 2
            spacy_model: spaCy model name for Tier 3
            use_gpu: Force GPU usage (None = auto-detect)
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect GPU availability
        if use_gpu is None:
            self.device = 0 if torch.cuda.is_available() else -1
            self.use_gpu = torch.cuda.is_available()
        else:
            self.device = 0 if use_gpu else -1
            self.use_gpu = use_gpu
        
        logger.info(f"Using device: {'GPU' if self.use_gpu else 'CPU'}")
        
        # Initialize tiers
        self.tier1_ner = None  # BERT-based
        self.tier2_ner = None  # Custom
        self.tier3_ner = None  # spaCy
        
        self._initialize_tier1(bert_model_name)
        self._initialize_tier2(custom_model_path)
        self._initialize_tier3(spacy_model)
        
        # Mapping for entity type normalization
        self.entity_type_mapping = {
            "PER": EntityType.PERSON.value,
            "PERSON": EntityType.PERSON.value,
            "LOC": EntityType.LOCATION.value,
            "LOCATION": EntityType.LOCATION.value,
            "GPE": EntityType.LOCATION.value,
            "ORG": "ORGANIZATION",
            "DATE": EntityType.DATE.value,
            "TIME": EntityType.TIME.value,
            "CARDINAL": EntityType.NUMBER.value,
            "QUANTITY": EntityType.NUMBER.value,
        }
    
    def _initialize_tier1(self, model_name: str):
        """Initialize Tier 1: BERT-based NER"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Tier 1 (BERT-NER) unavailable: transformers not installed")
            return
        
        try:
            logger.info(f"Loading Tier 1 model: {model_name}")
            self.tier1_ner = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=self.device
            )
            logger.info("✓ Tier 1 (BERT-NER) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 1: {e}")
            self.tier1_ner = None
    
    def _initialize_tier2(self, model_path: Optional[str]):
        """Initialize Tier 2: Custom fine-tuned model"""
        if model_path is None:
            logger.info("Tier 2 (Custom) not configured - skipping")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Tier 2 (Custom) unavailable: transformers not installed")
            return
        
        try:
            logger.info(f"Loading Tier 2 custom model from: {model_path}")
            self.tier2_ner = pipeline(
                "ner",
                model=model_path,
                tokenizer=model_path,
                aggregation_strategy="simple",
                device=self.device
            )
            logger.info("✓ Tier 2 (Custom) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 2: {e}")
            self.tier2_ner = None
    
    def _initialize_tier3(self, model_name: str):
        """Initialize Tier 3: spaCy NER"""
        if not SPACY_AVAILABLE:
            logger.warning("Tier 3 (spaCy) unavailable: spacy not installed")
            return
        
        try:
            logger.info(f"Loading Tier 3 model: {model_name}")
            
            # Try to load the model
            try:
                self.tier3_ner = spacy.load(model_name)
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found. Trying to download...")
                # Try with smaller model as fallback
                try:
                    self.tier3_ner = spacy.load("en_core_web_sm")
                    logger.info("✓ Tier 3 (spaCy) initialized with en_core_web_sm")
                except OSError:
                    logger.error("No spaCy models available. Install with: python -m spacy download en_core_web_sm")
                    self.tier3_ner = None
                    return
            
            logger.info("✓ Tier 3 (spaCy) initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tier 3: {e}")
            self.tier3_ner = None
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities with automatic fallback.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities
        """
        # Try Tier 1 (BERT-NER)
        if self.tier1_ner is not None:
            try:
                entities = self._extract_tier1(text)
                if entities:
                    logger.debug(f"Extracted {len(entities)} entities using Tier 1 (BERT)")
                    return entities
            except Exception as e:
                logger.warning(f"Tier 1 failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2 (Custom)
        if self.tier2_ner is not None:
            try:
                entities = self._extract_tier2(text)
                if entities:
                    logger.debug(f"Extracted {len(entities)} entities using Tier 2 (Custom)")
                    return entities
            except Exception as e:
                logger.warning(f"Tier 2 failed: {e}. Falling back to Tier 3...")
        
        # Try Tier 3 (spaCy)
        if self.tier3_ner is not None:
            try:
                entities = self._extract_tier3(text)
                logger.debug(f"Extracted {len(entities)} entities using Tier 3 (spaCy)")
                return entities
            except Exception as e:
                logger.error(f"All tiers failed. Last error: {e}")
        
        # All tiers failed
        logger.error("All entity extraction tiers failed")
        return []
    
    def _extract_tier1(self, text: str) -> List[Entity]:
        """Extract entities using Tier 1 (BERT-based)"""
        results = self.tier1_ner(text)
        entities = []
        
        for item in results:
            if item['score'] >= self.confidence_threshold:
                entity_type = self._normalize_entity_type(item['entity_group'])
                entities.append(Entity(
                    text=item['word'].strip(),
                    type=entity_type,
                    start=item['start'],
                    end=item['end'],
                    confidence=item['score'],
                    tier="Tier1-BERT"
                ))
        
        return entities
    
    def _extract_tier2(self, text: str) -> List[Entity]:
        """Extract entities using Tier 2 (Custom model)"""
        results = self.tier2_ner(text)
        entities = []
        
        for item in results:
            if item['score'] >= self.confidence_threshold:
                entity_type = self._normalize_entity_type(item['entity_group'])
                entities.append(Entity(
                    text=item['word'].strip(),
                    type=entity_type,
                    start=item['start'],
                    end=item['end'],
                    confidence=item['score'],
                    tier="Tier2-Custom"
                ))
        
        return entities
    
    def _extract_tier3(self, text: str) -> List[Entity]:
        """Extract entities using Tier 3 (spaCy)"""
        doc = self.tier3_ner(text)
        entities = []
        
        for ent in doc.ents:
            entity_type = self._normalize_entity_type(ent.label_)
            entities.append(Entity(
                text=ent.text,
                type=entity_type,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85,  # spaCy doesn't provide confidence scores by default
                tier="Tier3-spaCy"
            ))
        
        return entities
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """Normalize entity type to standard format"""
        return self.entity_type_mapping.get(entity_type.upper(), entity_type)
    
    def extract_by_type(self, text: str, entity_type: str) -> List[Entity]:
        """
        Extract entities of a specific type.
        
        Args:
            text: Input text
            entity_type: Specific entity type to extract
            
        Returns:
            List of entities matching the type
        """
        all_entities = self.extract(text)
        return [e for e in all_entities if e.type.upper() == entity_type.upper()]
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all tiers"""
        return {
            "tier1_bert": self.tier1_ner is not None,
            "tier2_custom": self.tier2_ner is not None,
            "tier3_spacy": self.tier3_ner is not None,
            "gpu_available": self.use_gpu,
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize extractor
    extractor = EntityExtractor()
    
    # Test sentences
    test_sentences = [
        "Bring me the red cup from the kitchen",
        "Navigate to the living room and find John",
        "What's the temperature sensor reading?",
        "Move forward 3 meters and turn left"
    ]
    
    print("=" * 80)
    print("ENTITY EXTRACTOR - TESTING")
    print("=" * 80)
    print(f"\nStatus: {extractor.get_status()}\n")
    
    for sentence in test_sentences:
        print(f"\nInput: {sentence}")
        entities = extractor.extract(sentence)
        
        if entities:
            for entity in entities:
                print(f"  → {entity.type:15} | {entity.text:20} | "
                      f"Confidence: {entity.confidence:.2f} | Tier: {entity.tier}")
        else:
            print("  → No entities found")
    
    print("\n" + "=" * 80)

