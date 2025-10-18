"""
Emotion Detection Module
Provides multi-tier emotion detection with automatic fallback
"""

from .detector import EmotionDetector, EmotionResult, EmotionLabel

__all__ = ['EmotionDetector', 'EmotionResult', 'EmotionLabel']

