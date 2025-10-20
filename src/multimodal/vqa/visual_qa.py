"""
Visual Question Answering (VQA) System

PURPOSE:
    Implements Visual Question Answering for humanoid robot applications.
    Enables the robot to answer questions about visual scenes using both
    vision and language understanding.

PIPELINE CONTEXT:
    
    VQA Flow:
    Question + Image → Vision Features → Language Features → Fusion → Answer
         ↓               ↓                ↓                ↓        ↓
    ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ ┌─────────┐
    │ "What   │ │ CLIP        │ │ BERT        │ │ Cross-  │ │ "A red  │
    │ color   │ │ Vision      │ │ Language    │ │ Modal   │ │ cup and │
    │ is the  │ │ Encoder     │ │ Encoder     │ │ Fusion  │ │ a blue  │
    │ cup?"   │ │ Object      │ │ Question    │ │ Network │ │ bottle" │
    └─────────┘ └─────────────┘ └─────────────┘ └─────────┘ └─────────┘

WHY VQA MATTERS:
    Current System: Separate vision and language processing
    With VQA: Unified understanding of visual scenes through language
    
    Benefits:
    - Natural scene understanding
    - Object counting and identification
    - Spatial relationship queries
    - Attribute-based questions
    - Context-aware responses

HOW IT WORKS:
    1. Vision Processing: Extract visual features from image
    2. Language Processing: Encode question into language features
    3. Fusion: Combine vision and language features
    4. Answer Generation: Generate natural language answer
    5. Confidence Scoring: Assess answer confidence

INTEGRATION WITH EXISTING SYSTEM:
    - Uses existing vision components (object detection, scene understanding)
    - Uses existing NLP components (language understanding, generation)
    - Adds VQA-specific fusion and answer generation
    - Enables natural scene interrogation

RELATED FILES:
    - src/vision/vision_service.py: Visual processing
    - src/nlp/nlp_service.py: Language processing
    - src/agents/multimodal_fusion.py: Multimodal fusion
    - src/multimodal/grounding/visual_grounding.py: Visual grounding

USAGE:
    # Initialize VQA system
    vqa = VisualQuestionAnswering(config)
    
    # Ask questions about an image
    result = await vqa.answer_question(
        image=camera_image,
        question="What objects do you see?"
    )
    
    # Get answer
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
import time
from enum import Enum

# Import existing modules
from src.vision.vision_service import VisionService
from src.nlp.nlp_service import NLPService
from src.agents.multimodal_fusion import MultimodalFusion

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of VQA questions"""
    OBJECT_COUNT = "object_count"
    OBJECT_IDENTIFICATION = "object_identification"
    ATTRIBUTE_QUERY = "attribute_query"
    SPATIAL_QUERY = "spatial_query"
    SCENE_DESCRIPTION = "scene_description"
    YES_NO = "yes_no"
    COLOR_QUERY = "color_query"
    SIZE_QUERY = "size_query"
    ACTION_QUERY = "action_query"
    GENERAL = "general"


@dataclass
class VQAResult:
    """Result from Visual Question Answering"""
    answer: str
    confidence: float
    question_type: QuestionType
    reasoning: List[str]
    visual_evidence: Dict[str, Any]
    language_evidence: Dict[str, Any]
    fusion_confidence: float
    timestamp: float


@dataclass
class VQAQuestion:
    """Parsed VQA question"""
    original_question: str
    question_type: QuestionType
    target_objects: List[str]
    attributes: List[str]
    spatial_relations: List[str]
    expected_answer_type: str  # "yes_no", "count", "object", "attribute", "description"
    confidence: float


class VQAAnswerGenerator:
    """
    Neural network for generating VQA answers
    """
    
    def __init__(self, 
                 vision_dim: int = 512,
                 language_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 1000,
                 max_answer_length: int = 20):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_answer_length = max_answer_length
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Answer generation head
        self.answer_head = nn.Linear(hidden_dim, vocab_size)
        
        # Question type classification head
        self.type_head = nn.Linear(hidden_dim, len(QuestionType))
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VQA answer generation
        
        Args:
            vision_features: [batch, vision_dim]
            language_features: [batch, language_dim]
        
        Returns:
            Dictionary with answer logits, question type, and confidence
        """
        # Concatenate features
        combined = torch.cat([vision_features, language_features], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Generate outputs
        answer_logits = self.answer_head(fused)
        type_logits = self.type_head(fused)
        confidence = self.confidence_head(fused)
        
        return {
            'answer_logits': answer_logits,
            'type_logits': type_logits,
            'confidence': confidence
        }


class VisualQuestionAnswering:
    """
    Visual Question Answering system for humanoid robots
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VQA system
        
        Args:
            config: Configuration for VQA models and parameters
        """
        self.config = config
        
        # Initialize components
        self.vision_service = VisionService(config.get('vision', {}))
        self.nlp_service = NLPService(config.get('nlp', {}))
        self.multimodal_fusion = MultimodalFusion(config.get('multimodal', {}))
        
        # VQA parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.max_answer_length = config.get('max_answer_length', 20)
        
        # Initialize answer generator
        self.answer_generator = VQAAnswerGenerator(
            vision_dim=config.get('vision_dim', 512),
            language_dim=config.get('language_dim', 768),
            hidden_dim=config.get('hidden_dim', 512),
            vocab_size=config.get('vocab_size', 1000),
            max_answer_length=self.max_answer_length
        )
        
        # VQA vocabulary (simplified)
        self.vocab = self._build_vocab()
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        # Question patterns for classification
        self.question_patterns = self._build_question_patterns()
        
        logger.info("Visual Question Answering system initialized")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for answer generation"""
        vocab = {
            '<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3,
            'yes': 4, 'no': 5, 'zero': 6, 'one': 7, 'two': 8, 'three': 9, 'four': 10, 'five': 11,
            'cup': 12, 'bottle': 13, 'phone': 14, 'book': 15, 'remote': 16, 'keys': 17,
            'person': 18, 'hand': 19, 'face': 20, 'chair': 21, 'table': 22,
            'red': 23, 'blue': 24, 'green': 25, 'yellow': 26, 'black': 27, 'white': 28,
            'small': 29, 'large': 30, 'big': 31, 'tiny': 32,
            'left': 33, 'right': 34, 'center': 35, 'top': 36, 'bottom': 37,
            'kitchen': 38, 'living': 39, 'room': 40, 'office': 41,
            'i': 42, 'see': 43, 'a': 44, 'the': 45, 'and': 46, 'in': 47, 'on': 48, 'at': 49
        }
        
        # Add more words
        for i in range(50, 1000):
            vocab[f'word_{i}'] = i
        
        return vocab
    
    def _build_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Build patterns for question type classification"""
        return {
            QuestionType.OBJECT_COUNT: [
                r'\bhow many\b', r'\bcount\b', r'\bnumber of\b'
            ],
            QuestionType.OBJECT_IDENTIFICATION: [
                r'\bwhat\b.*\b(?:is|are)\b', r'\bidentify\b', r'\bname\b'
            ],
            QuestionType.ATTRIBUTE_QUERY: [
                r'\bwhat color\b', r'\bwhat size\b', r'\bhow big\b', r'\bhow small\b'
            ],
            QuestionType.SPATIAL_QUERY: [
                r'\bwhere\b', r'\bposition\b', r'\blocation\b', r'\b(?:left|right|top|bottom)\b'
            ],
            QuestionType.SCENE_DESCRIPTION: [
                r'\bdescribe\b', r'\bwhat do you see\b', r'\btell me about\b'
            ],
            QuestionType.YES_NO: [
                r'\bis there\b', r'\bdo you see\b', r'\bcan you see\b', r'\bis that\b'
            ],
            QuestionType.COLOR_QUERY: [
                r'\bcolor\b', r'\bwhat color\b', r'\bcolored\b'
            ],
            QuestionType.SIZE_QUERY: [
                r'\bsize\b', r'\bbig\b', r'\bsmall\b', r'\blarge\b', r'\btiny\b'
            ],
            QuestionType.ACTION_QUERY: [
                r'\bwhat is.*doing\b', r'\baction\b', r'\bactivity\b'
            ]
        }
    
    async def answer_question(self, 
                            image: np.ndarray,
                            question: str) -> VQAResult:
        """
        Answer a question about an image
        
        Args:
            image: Input image
            question: Natural language question
        
        Returns:
            VQA result with answer and confidence
        """
        try:
            logger.info(f"Answering question: '{question}'")
            
            # 1. Parse question
            parsed_question = self._parse_question(question)
            logger.info(f"Question type: {parsed_question.question_type}")
            
            # 2. Process image
            vision_result = await self.vision_service.process_image(image)
            
            # 3. Process question
            language_result = await self.nlp_service.process_text(question)
            
            # 4. Extract features
            vision_features = await self._extract_vision_features(vision_result)
            language_features = await self._extract_language_features(language_result)
            
            # 5. Generate answer
            answer_result = await self._generate_answer(
                vision_features, language_features, parsed_question
            )
            
            # 6. Post-process answer
            final_answer = self._post_process_answer(answer_result, parsed_question)
            
            # 7. Calculate confidence
            confidence = self._calculate_confidence(answer_result, vision_result, language_result)
            
            # 8. Generate reasoning
            reasoning = self._generate_reasoning(parsed_question, vision_result, answer_result)
            
            return VQAResult(
                answer=final_answer,
                confidence=confidence,
                question_type=parsed_question.question_type,
                reasoning=reasoning,
                visual_evidence=vision_result,
                language_evidence=language_result,
                fusion_confidence=answer_result.get('confidence', 0.0),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return VQAResult(
                answer="I'm sorry, I couldn't answer that question.",
                confidence=0.0,
                question_type=QuestionType.GENERAL,
                reasoning=[f"Error: {str(e)}"],
                visual_evidence={},
                language_evidence={},
                fusion_confidence=0.0,
                timestamp=time.time()
            )
    
    def _parse_question(self, question: str) -> VQAQuestion:
        """
        Parse natural language question
        
        Args:
            question: Natural language question
        
        Returns:
            Parsed question with type and components
        """
        question_lower = question.lower().strip()
        
        # Determine question type
        question_type = self._classify_question_type(question_lower)
        
        # Extract target objects
        target_objects = self._extract_target_objects(question_lower)
        
        # Extract attributes
        attributes = self._extract_attributes(question_lower)
        
        # Extract spatial relations
        spatial_relations = self._extract_spatial_relations(question_lower)
        
        # Determine expected answer type
        expected_answer_type = self._determine_answer_type(question_type, question_lower)
        
        # Calculate parsing confidence
        confidence = self._calculate_parsing_confidence(question_type, target_objects, attributes)
        
        return VQAQuestion(
            original_question=question,
            question_type=question_type,
            target_objects=target_objects,
            attributes=attributes,
            spatial_relations=spatial_relations,
            expected_answer_type=expected_answer_type,
            confidence=confidence
        )
    
    def _classify_question_type(self, question: str) -> QuestionType:
        """Classify question type based on patterns"""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question):
                    return q_type
        
        return QuestionType.GENERAL
    
    def _extract_target_objects(self, question: str) -> List[str]:
        """Extract target objects from question"""
        objects = []
        
        # Common object names
        object_names = [
            'cup', 'bottle', 'phone', 'book', 'remote', 'keys', 'pen', 'pencil',
            'laptop', 'mouse', 'keyboard', 'chair', 'table', 'person', 'hand',
            'face', 'car', 'truck', 'bike', 'ball', 'toy', 'food', 'plate',
            'bowl', 'spoon', 'fork', 'knife', 'glass', 'mug', 'bag', 'box'
        ]
        
        for obj in object_names:
            if obj in question:
                objects.append(obj)
        
        return objects
    
    def _extract_attributes(self, question: str) -> List[str]:
        """Extract attributes from question"""
        attributes = []
        
        # Color attributes
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'brown']
        for color in colors:
            if color in question:
                attributes.append(f"color:{color}")
        
        # Size attributes
        sizes = ['small', 'large', 'big', 'tiny', 'huge', 'medium']
        for size in sizes:
            if size in question:
                attributes.append(f"size:{size}")
        
        return attributes
    
    def _extract_spatial_relations(self, question: str) -> List[str]:
        """Extract spatial relations from question"""
        relations = []
        
        spatial_words = ['left', 'right', 'top', 'bottom', 'center', 'middle', 'front', 'back', 'near', 'far']
        for word in spatial_words:
            if word in question:
                relations.append(word)
        
        return relations
    
    def _determine_answer_type(self, question_type: QuestionType, question: str) -> str:
        """Determine expected answer type"""
        if question_type == QuestionType.YES_NO:
            return "yes_no"
        elif question_type == QuestionType.OBJECT_COUNT:
            return "count"
        elif question_type == QuestionType.OBJECT_IDENTIFICATION:
            return "object"
        elif question_type == QuestionType.ATTRIBUTE_QUERY:
            return "attribute"
        elif question_type == QuestionType.SCENE_DESCRIPTION:
            return "description"
        else:
            return "general"
    
    def _calculate_parsing_confidence(self, question_type: QuestionType, target_objects: List[str], attributes: List[str]) -> float:
        """Calculate confidence in question parsing"""
        confidence = 0.5  # Base confidence
        
        if question_type != QuestionType.GENERAL:
            confidence += 0.2
        
        if target_objects:
            confidence += 0.1 * min(len(target_objects), 3)
        
        if attributes:
            confidence += 0.1 * min(len(attributes), 2)
        
        return min(confidence, 1.0)
    
    async def _extract_vision_features(self, vision_result: Dict[str, Any]) -> np.ndarray:
        """Extract vision features from vision service result"""
        # Use object detection features
        objects = vision_result.get('objects', [])
        
        # Create feature vector based on detected objects
        feature_vector = np.zeros(512)  # Fixed size for simplicity
        
        if objects:
            # Encode object information
            for i, obj in enumerate(objects[:10]):  # Limit to 10 objects
                obj_class = obj.get('class', 'unknown')
                confidence = obj.get('confidence', 0.0)
                
                # Simple encoding (would be more sophisticated in production)
                feature_vector[i * 50:(i + 1) * 50] = confidence
        
        return feature_vector
    
    async def _extract_language_features(self, language_result: Dict[str, Any]) -> np.ndarray:
        """Extract language features from NLP service result"""
        # Use intent and entity features
        intent = language_result.get('intent', '')
        entities = language_result.get('entities', [])
        
        # Create feature vector
        feature_vector = np.zeros(768)  # BERT-like size
        
        # Simple encoding (would use proper embeddings in production)
        if intent:
            feature_vector[:100] = 0.5  # Intent encoding
        
        if entities:
            feature_vector[100:200] = min(len(entities) / 10.0, 1.0)  # Entity count
        
        return feature_vector
    
    async def _generate_answer(self, 
                             vision_features: np.ndarray,
                             language_features: np.ndarray,
                             parsed_question: VQAQuestion) -> Dict[str, Any]:
        """Generate answer using neural network"""
        try:
            # Convert to tensors
            vision_tensor = torch.tensor(vision_features).unsqueeze(0).float()
            language_tensor = torch.tensor(language_features).unsqueeze(0).float()
            
            # Forward pass
            with torch.no_grad():
                output = self.answer_generator(vision_tensor, language_tensor)
            
            # Extract results
            answer_logits = output['answer_logits']
            type_logits = output['type_logits']
            confidence = output['confidence']
            
            # Get predicted answer
            predicted_indices = torch.argmax(answer_logits, dim=1)
            predicted_words = [self.idx_to_word.get(idx.item(), '<unk>') for idx in predicted_indices]
            
            return {
                'answer_logits': answer_logits.cpu().numpy(),
                'predicted_words': predicted_words,
                'confidence': confidence.item(),
                'type_logits': type_logits.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                'answer_logits': np.zeros((1, len(self.vocab))),
                'predicted_words': ['<unk>'],
                'confidence': 0.0,
                'type_logits': np.zeros((1, len(QuestionType)))
            }
    
    def _post_process_answer(self, 
                           answer_result: Dict[str, Any],
                           parsed_question: VQAQuestion) -> str:
        """Post-process generated answer"""
        predicted_words = answer_result.get('predicted_words', ['<unk>'])
        
        # Filter out special tokens
        filtered_words = [word for word in predicted_words if word not in ['<pad>', '<unk>', '<start>', '<end>']]
        
        if not filtered_words:
            return "I don't know."
        
        # Generate answer based on question type
        if parsed_question.question_type == QuestionType.YES_NO:
            if 'yes' in filtered_words:
                return "Yes"
            elif 'no' in filtered_words:
                return "No"
            else:
                return "I'm not sure."
        
        elif parsed_question.question_type == QuestionType.OBJECT_COUNT:
            # Look for numbers
            numbers = ['zero', 'one', 'two', 'three', 'four', 'five']
            for word in filtered_words:
                if word in numbers:
                    return word.capitalize()
            return "I can't count the objects."
        
        elif parsed_question.question_type == QuestionType.OBJECT_IDENTIFICATION:
            # Look for object names
            objects = ['cup', 'bottle', 'phone', 'book', 'remote', 'keys', 'person', 'chair', 'table']
            for word in filtered_words:
                if word in objects:
                    return f"I see a {word}."
            return "I can't identify the object."
        
        elif parsed_question.question_type == QuestionType.ATTRIBUTE_QUERY:
            # Look for attributes
            colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
            sizes = ['small', 'large', 'big', 'tiny']
            
            for word in filtered_words:
                if word in colors:
                    return f"It's {word}."
                elif word in sizes:
                    return f"It's {word}."
            return "I can't determine the attribute."
        
        else:
            # General answer
            return " ".join(filtered_words[:5]).capitalize()
    
    def _calculate_confidence(self, 
                            answer_result: Dict[str, Any],
                            vision_result: Dict[str, Any],
                            language_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in the answer"""
        # Base confidence from answer generation
        base_confidence = answer_result.get('confidence', 0.0)
        
        # Adjust based on vision quality
        vision_objects = vision_result.get('objects', [])
        if vision_objects:
            avg_vision_confidence = sum(obj.get('confidence', 0.0) for obj in vision_objects) / len(vision_objects)
            base_confidence = (base_confidence + avg_vision_confidence) / 2
        
        # Adjust based on language quality
        language_confidence = language_result.get('confidence', 0.5)
        base_confidence = (base_confidence + language_confidence) / 2
        
        return min(base_confidence, 1.0)
    
    def _generate_reasoning(self, 
                          parsed_question: VQAQuestion,
                          vision_result: Dict[str, Any],
                          answer_result: Dict[str, Any]) -> List[str]:
        """Generate reasoning for the answer"""
        reasoning = []
        
        reasoning.append(f"Question type: {parsed_question.question_type.value}")
        
        if parsed_question.target_objects:
            reasoning.append(f"Looking for objects: {', '.join(parsed_question.target_objects)}")
        
        if parsed_question.attributes:
            reasoning.append(f"Checking attributes: {', '.join(parsed_question.attributes)}")
        
        vision_objects = vision_result.get('objects', [])
        if vision_objects:
            reasoning.append(f"Detected {len(vision_objects)} objects in the scene")
            for obj in vision_objects[:3]:  # Show top 3 objects
                reasoning.append(f"  - {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0.0):.2f})")
        
        reasoning.append(f"Answer confidence: {answer_result.get('confidence', 0.0):.2f}")
        
        return reasoning
    
    async def batch_answer_questions(self, 
                                   image: np.ndarray,
                                   questions: List[str]) -> List[VQAResult]:
        """
        Answer multiple questions about the same image
        
        Args:
            image: Input image
            questions: List of questions
        
        Returns:
            List of VQA results
        """
        results = []
        
        for question in questions:
            result = await self.answer_question(image, question)
            results.append(result)
        
        return results
    
    def get_vqa_statistics(self) -> Dict[str, Any]:
        """Get VQA system statistics"""
        return {
            'vocab_size': len(self.vocab),
            'max_answer_length': self.max_answer_length,
            'confidence_threshold': self.confidence_threshold,
            'supported_question_types': [q_type.value for q_type in QuestionType],
            'timestamp': time.time()
        }


# Example usage and testing
async def main():
    """Example usage of Visual Question Answering"""
    config = {
        'vision': {
            'object_detection': {'enabled': True},
            'scene_understanding': {'enabled': True}
        },
        'nlp': {
            'intent_classifier': {'enabled': True},
            'entity_extractor': {'enabled': True}
        },
        'multimodal': {
            'vision_dim': 512,
            'text_dim': 768,
            'audio_dim': 256
        },
        'confidence_threshold': 0.3,
        'max_answer_length': 20
    }
    
    # Initialize VQA system
    vqa = VisualQuestionAnswering(config)
    
    # Test with synthetic image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test questions
    questions = [
        "What objects do you see?",
        "How many objects are there?",
        "What color is the cup?",
        "Is there a person in the image?",
        "Where is the bottle?",
        "Describe what you see"
    ]
    
    for question in questions:
        result = await vqa.answer_question(test_image, question)
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Type: {result.question_type.value}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 50)
    
    # Test batch questions
    batch_results = await vqa.batch_answer_questions(test_image, questions)
    print(f"Batch VQA completed: {len(batch_results)} results")
    
    # Get statistics
    stats = vqa.get_vqa_statistics()
    print(f"VQA statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
