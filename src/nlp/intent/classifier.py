"""
Intent Classifier - Evolved from my previous work onchapo-bot with transformer-based classification

This module provides robust intent classification using:
1. Transformer-based model (BERT/RoBERTa) for high accuracy
2. Rule-based fallback for safety-critical intents
3. Intent normalization and aliasing
4. Confidence calibration

Author: Victor Ibhafidon
Organization: Xtainless Technologies
Date: 2025-10-09
"""

import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError:
    logging.warning("ONNX Runtime or Transformers not installed. Intent classifier will use fallback mode.")
    ort = None


class IntentCategory(Enum):
    """Intent categories for the humanoid robot"""
    # Navigation & Mobility
    NAVIGATE_TO = "navigate_to"
    FOLLOW_ME = "follow_me"
    STOP_MOVING = "stop_moving"
    COME_HERE = "come_here"
    
    # Manipulation
    FETCH_OBJECT = "fetch_object"
    PLACE_OBJECT = "place_object"
    GRASP = "grasp"
    RELEASE = "release"
    HAND_OVER = "hand_over"
    
    # Perception Queries
    WHAT_DO_YOU_SEE = "what_do_you_see"
    FIND_OBJECT = "find_object"
    IDENTIFY_PERSON = "identify_person"
    WHERE_IS = "where_is"
    
    # Information Queries
    GET_WEATHER = "get_weather"
    GET_NEWS = "get_news"
    TELL_TIME = "tell_time"
    ANSWER_QUESTION = "answer_question"
    
    # Social Interaction
    GREETING = "greeting"
    GOODBYE = "goodbye"
    SMALL_TALK = "small_talk"
    HOW_ARE_YOU = "how_are_you"
    TELL_JOKE = "tell_joke"
    
    # Task Management
    SET_REMINDER = "set_reminder"
    LIST_REMINDERS = "list_reminders"
    DELETE_REMINDER = "delete_reminder"
    SET_ALARM = "set_alarm"
    STOP_ALARM = "stop_alarm"
    
    # System Control
    HELP = "help"
    STATUS = "status"
    SHUTDOWN = "shutdown"
    EMERGENCY_STOP = "emergency_stop"
    
    # Emotion & Affective
    EMOTION_EXPRESSION = "emotion_expression"
    COMFORT_REQUEST = "comfort_request"
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class IntentPrediction:
    """Container for intent prediction results"""
    intent: IntentCategory
    confidence: float
    is_safety_critical: bool
    used_fallback: bool
    normalized_from: Optional[str] = None
    top_k_predictions: Optional[List[Tuple[str, float]]] = None


# Intent normalization map (evolved from chapo-bot)
INTENT_NORMALIZATION_MAP = {
    # Navigation
    "go_to": "navigate_to",
    "move_to": "navigate_to",
    "travel_to": "navigate_to",
    "walk_to": "navigate_to",
    "follow": "follow_me",
    "follow_user": "follow_me",
    "stop": "emergency_stop",  # Safety-critical
    "halt": "emergency_stop",
    "freeze": "emergency_stop",
    
    # Manipulation
    "bring": "fetch_object",
    "get": "fetch_object",
    "retrieve": "fetch_object",
    "pick_up": "grasp",
    "grab": "grasp",
    "take": "grasp",
    "put_down": "place_object",
    "drop": "release",
    "let_go": "release",
    "give_me": "hand_over",
    
    # Perception
    "what_can_you_see": "what_do_you_see",
    "look_for": "find_object",
    "search_for": "find_object",
    "locate": "find_object",
    "who_is_that": "identify_person",
    
    # Information
    "weather": "get_weather",
    "forecast": "get_weather",
    "weather_forecast": "get_weather",
    "news": "get_news",
    "headlines": "get_news",
    "latest_news": "get_news",
    "what_time": "tell_time",
    "time_now": "tell_time",
    "current_time": "tell_time",
    
    # Social
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "bye": "goodbye",
    "goodbye": "goodbye",
    "see_you": "goodbye",
    "how_are_you": "how_are_you",
    "idle_convo": "small_talk",
    "casual_chat": "small_talk",
    "joke": "tell_joke",
    "funny": "tell_joke",
    
    # Task Management
    "remind_me": "set_reminder",
    "reminder": "set_reminder",
    "alarm": "set_alarm",
    "wake_me": "set_alarm",
    "cancel_alarm": "stop_alarm",
    "delete_alarm": "stop_alarm",
    
    # System
    "what_can_you_do": "help",
    "capabilities": "status",
    "system_status": "status",
    "turn_off": "shutdown",
    "power_off": "shutdown",
}

# Safety-critical intents that bypass normal processing
SAFETY_CRITICAL_INTENTS = {
    IntentCategory.EMERGENCY_STOP,
    IntentCategory.STOP_MOVING,
    IntentCategory.RELEASE,  # Drop object immediately
}

# Rule-based patterns for safety-critical intents (deterministic)
SAFETY_RULES = {
    IntentCategory.EMERGENCY_STOP: [
        r"\b(stop|halt|freeze|emergency)\b",
        r"\b(e-?stop|estop)\b",
    ],
    IntentCategory.RELEASE: [
        r"\b(drop|release|let go)\b.*\b(now|immediately|right now)\b",
    ],
}


class IntentClassifier:
    """
    Hybrid intent classifier using transformers + rule-based fallback.
    
    Prioritizes safety by using deterministic rules for critical intents.
    Uses confidence calibration to provide reliable uncertainty estimates.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_name: str = "bert-base-uncased",
        confidence_threshold: float = 0.7,
        use_rules_fallback: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the intent classifier.
        
        Args:
            model_path: Path to ONNX model file (if None, uses rules only)
            tokenizer_name: Hugging Face tokenizer name
            confidence_threshold: Minimum confidence for non-rule-based predictions
            use_rules_fallback: Whether to use rule-based fallback
            device: Device for inference ("cpu", "cuda", "tensorrt")
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.use_rules_fallback = use_rules_fallback
        
        # Load transformer model
        self.model = None
        self.tokenizer = None
        if model_path and ort:
            try:
                self.model = ort.InferenceSession(
                    model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    if device != "cpu" else ['CPUExecutionProvider']
                )
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.logger.info(f"Loaded intent classifier from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}. Using rule-based only.")
        
        # Intent label mapping (should match training)
        self.label_to_intent = {i: intent.value for i, intent in enumerate(IntentCategory)}
        
        self.logger.info(f"Intent classifier initialized (model={'loaded' if self.model else 'rules-only'})")
    
    def normalize_intent(self, intent_str: str) -> Tuple[str, Optional[str]]:
        """
        Normalize intent string using alias map.
        
        Args:
            intent_str: Raw intent string
        
        Returns:
            Tuple of (normalized_intent, original_intent)
        """
        intent_lower = intent_str.lower().strip()
        normalized = INTENT_NORMALIZATION_MAP.get(intent_lower, intent_str)
        original = intent_str if normalized != intent_str else None
        return normalized, original
    
    def _check_safety_rules(self, text: str) -> Optional[IntentCategory]:
        """
        Check if text matches safety-critical patterns.
        
        Args:
            text: User input text
        
        Returns:
            IntentCategory if matched, None otherwise
        """
        import re
        text_lower = text.lower()
        
        for intent, patterns in SAFETY_RULES.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    self.logger.warning(f"Safety-critical intent detected: {intent.value}")
                    return intent
        return None
    
    def _predict_with_model(self, text: str, top_k: int = 3) -> List[Tuple[IntentCategory, float]]:
        """
        Predict intent using transformer model.
        
        Args:
            text: Input text
            top_k: Number of top predictions to return
        
        Returns:
            List of (intent, confidence) tuples
        """
        if not self.model or not self.tokenizer:
            return [(IntentCategory.UNKNOWN, 0.0)]
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="np",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            
            # Run inference
            outputs = self.model.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            
            # Get logits and apply softmax
            logits = outputs[0][0]
            probs = self._softmax(logits)
            
            # Get top-k predictions
            top_indices = np.argsort(probs)[-top_k:][::-1]
            predictions = [
                (IntentCategory(self.label_to_intent[idx]), float(probs[idx]))
                for idx in top_indices
            ]
            
            return predictions
        
        except Exception as e:
            self.logger.error(f"Model inference error: {e}")
            return [(IntentCategory.UNKNOWN, 0.0)]
    
    def _predict_with_rules(self, text: str) -> Tuple[IntentCategory, float]:
        """
        Simple rule-based intent classification (fallback).
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (intent, confidence)
        """
        text_lower = text.lower()
        
        # Check common patterns (adapted from chapo-bot)
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return IntentCategory.GREETING, 0.9
        
        if any(word in text_lower for word in ["bye", "goodbye", "see you", "later"]):
            return IntentCategory.GOODBYE, 0.9
        
        if "how are you" in text_lower:
            return IntentCategory.HOW_ARE_YOU, 0.9
        
        if any(word in text_lower for word in ["bring", "get", "fetch"]):
            return IntentCategory.FETCH_OBJECT, 0.7
        
        if any(word in text_lower for word in ["weather", "forecast"]):
            return IntentCategory.GET_WEATHER, 0.8
        
        if any(word in text_lower for word in ["news", "headlines"]):
            return IntentCategory.GET_NEWS, 0.8
        
        if any(word in text_lower for word in ["time", "clock"]):
            return IntentCategory.TELL_TIME, 0.8
        
        if any(word in text_lower for word in ["help", "what can you do"]):
            return IntentCategory.HELP, 0.85
        
        # Default
        return IntentCategory.SMALL_TALK, 0.5
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict(self, text: str, return_top_k: int = 1) -> IntentPrediction:
        """
        Predict intent from text input.
        
        Args:
            text: User input text
            return_top_k: Number of top predictions to include
        
        Returns:
            IntentPrediction object
        """
        # 1. Safety check (highest priority)
        safety_intent = self._check_safety_rules(text)
        if safety_intent:
            return IntentPrediction(
                intent=safety_intent,
                confidence=1.0,
                is_safety_critical=True,
                used_fallback=True,
                top_k_predictions=[(safety_intent.value, 1.0)]
            )
        
        # 2. Try model-based prediction
        predictions = []
        used_fallback = False
        
        if self.model:
            predictions = self._predict_with_model(text, top_k=return_top_k)
            top_intent, top_confidence = predictions[0]
            
            # Check confidence threshold
            if top_confidence < self.confidence_threshold and self.use_rules_fallback:
                self.logger.info(
                    f"Low confidence ({top_confidence:.2f}), using rule-based fallback"
                )
                top_intent, top_confidence = self._predict_with_rules(text)
                used_fallback = True
        else:
            # Model not available, use rules
            top_intent, top_confidence = self._predict_with_rules(text)
            predictions = [(top_intent, top_confidence)]
            used_fallback = True
        
        # 3. Normalize intent
        if isinstance(top_intent, str):
            normalized, original = self.normalize_intent(top_intent)
            try:
                top_intent = IntentCategory(normalized)
            except ValueError:
                top_intent = IntentCategory.UNKNOWN
        else:
            original = None
        
        # 4. Build result
        is_safety_critical = top_intent in SAFETY_CRITICAL_INTENTS
        
        return IntentPrediction(
            intent=top_intent,
            confidence=top_confidence,
            is_safety_critical=is_safety_critical,
            used_fallback=used_fallback,
            normalized_from=original,
            top_k_predictions=[(intent.value if hasattr(intent, 'value') else intent, conf) 
                             for intent, conf in predictions[:return_top_k]]
        )


# ==================== Standalone Testing ====================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize classifier (rule-based mode for demo)
    classifier = IntentClassifier(
        model_path=None,  # Set to actual path for transformer mode
        use_rules_fallback=True
    )
    
    # Test cases
    test_inputs = [
        "STOP RIGHT NOW",  # Safety-critical
        "Hello! How are you?",
        "Can you bring me a cup of water?",
        "What's the weather like today?",
        "Tell me the latest news",
        "What time is it?",
        "Fetch the red ball from the kitchen",
        "EMERGENCY STOP",
        "Help me understand what you can do",
        "I'm feeling sad today",
    ]
    
    print("\n" + "="*70)
    print("Intent Classifier Test")
    print("="*70 + "\n")
    
    for text in test_inputs:
        prediction = classifier.predict(text, return_top_k=3)
        
        print(f"Input: {text}")
        print(f"  Intent: {prediction.intent.value}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Safety Critical: {prediction.is_safety_critical}")
        print(f"  Used Fallback: {prediction.used_fallback}")
        if prediction.top_k_predictions and len(prediction.top_k_predictions) > 1:
            print(f"  Top-3: {prediction.top_k_predictions}")
        print()

