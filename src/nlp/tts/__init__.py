"""
TTS (Text-to-Speech) Module
Provides multi-tier speech synthesis with ElevenLabs, Coqui, and pyttsx3
"""

from .synthesizer import TTSSynthesizer, TTSResult

__all__ = ['TTSSynthesizer', 'TTSResult']

