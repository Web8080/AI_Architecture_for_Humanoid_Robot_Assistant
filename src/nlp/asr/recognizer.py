"""
ASR (Automatic Speech Recognition) with 2-Tier Fallback
Tier 1: Whisper (OpenAI Whisper via faster-whisper)
Tier 2: Vosk (Lightweight, streaming capable)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time

# Tier 1: Faster-Whisper (optimized Whisper.cpp Python wrapper)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not available. Install with: pip install faster-whisper")

# Alternative: OpenAI Whisper (if faster-whisper not available)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("openai-whisper not available. Install with: pip install openai-whisper")

# Tier 2: Vosk
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
    import json as json_lib
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available. Install with: pip install vosk")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """Represents ASR result"""
    text: str
    confidence: float
    language: str
    tier: str
    latency_ms: float
    segments: Optional[List[Dict]] = None


class ASRRecognizer:
    """
    Multi-tier ASR with automatic fallback.
    Tier 1: Whisper (best accuracy)
    Tier 2: Vosk (lightweight, streaming)
    Auto-detects GPU/CPU and selects appropriate model size.
    """
    
    def __init__(
        self,
        whisper_model_size: str = "base",  # tiny, base, small, medium, large
        vosk_model_path: Optional[str] = None,
        language: str = "en",
        use_gpu: bool = None,
        beam_size: int = 5
    ):
        """
        Initialize ASR recognizer.
        
        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large)
            vosk_model_path: Path to Vosk model directory
            language: Language code (en, es, fr, etc.)
            use_gpu: Force GPU usage (None = auto-detect)
            beam_size: Beam search size for Whisper
        """
        self.whisper_model_size = whisper_model_size
        self.vosk_model_path = vosk_model_path
        self.language = language
        self.beam_size = beam_size
        
        # Auto-detect GPU
        if use_gpu is None:
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
            except ImportError:
                self.use_gpu = False
        else:
            self.use_gpu = use_gpu
        
        # Select compute type based on GPU
        if self.use_gpu:
            self.compute_type = "float16"
            self.device = "cuda"
        else:
            self.compute_type = "int8"
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device} with compute type: {self.compute_type}")
        
        # Initialize tiers
        self.tier1_model = None  # Whisper
        self.tier2_model = None  # Vosk
        
        self._initialize_tier1()
        self._initialize_tier2()
    
    def _initialize_tier1(self):
        """Initialize Tier 1: Whisper"""
        # Try faster-whisper first (preferred)
        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Tier 1 (Faster-Whisper): {self.whisper_model_size}")
                self.tier1_model = WhisperModel(
                    self.whisper_model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
                logger.info("✓ Tier 1 (Faster-Whisper) initialized successfully")
                self.tier1_type = "faster-whisper"
                return
            except Exception as e:
                logger.warning(f"Faster-Whisper failed: {e}")
        
        # Fallback to regular Whisper
        if WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Tier 1 (OpenAI Whisper): {self.whisper_model_size}")
                self.tier1_model = whisper.load_model(self.whisper_model_size, device=self.device)
                logger.info("✓ Tier 1 (OpenAI Whisper) initialized successfully")
                self.tier1_type = "openai-whisper"
                return
            except Exception as e:
                logger.error(f"OpenAI Whisper failed: {e}")
        
        logger.warning("Tier 1 (Whisper) unavailable")
        self.tier1_model = None
    
    def _initialize_tier2(self):
        """Initialize Tier 2: Vosk"""
        if not VOSK_AVAILABLE:
            logger.warning("Tier 2 (Vosk) unavailable: vosk not installed")
            return
        
        if self.vosk_model_path is None:
            logger.warning("Tier 2 (Vosk) unavailable: no model path provided")
            logger.info("  Download models from: https://alphacephei.com/vosk/models")
            return
        
        try:
            logger.info(f"Loading Tier 2 (Vosk) from: {self.vosk_model_path}")
            self.tier2_model = VoskModel(self.vosk_model_path)
            logger.info("✓ Tier 2 (Vosk) initialized successfully")
        except Exception as e:
            logger.error(f"Vosk initialization failed: {e}")
            self.tier2_model = None
    
    def transcribe(self, audio_path: str) -> ASRResult:
        """
        Transcribe audio file with automatic fallback.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            ASR result
        """
        start_time = time.time()
        
        # Try Tier 1: Whisper
        if self.tier1_model is not None:
            try:
                result = self._transcribe_whisper(audio_path)
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Transcribed using Tier 1: {result.text[:50]}...")
                return result
            except Exception as e:
                logger.warning(f"Tier 1 (Whisper) failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2: Vosk
        if self.tier2_model is not None:
            try:
                result = self._transcribe_vosk(audio_path)
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Transcribed using Tier 2: {result.text[:50]}...")
                return result
            except Exception as e:
                logger.error(f"All ASR tiers failed. Last error: {e}")
        
        # All failed
        return ASRResult(
            text="[Transcription failed]",
            confidence=0.0,
            language="unknown",
            tier="Failed",
            latency_ms=(time.time() - start_time) * 1000
        )
    
    def _transcribe_whisper(self, audio_path: str) -> ASRResult:
        """Transcribe using Whisper"""
        if self.tier1_type == "faster-whisper":
            # Use faster-whisper
            segments, info = self.tier1_model.transcribe(
                audio_path,
                language=self.language,
                beam_size=self.beam_size
            )
            
            # Collect segments
            segments_list = []
            full_text = []
            avg_confidence = 0.0
            
            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.avg_logprob
                })
                full_text.append(segment.text)
                avg_confidence += segment.avg_logprob
            
            if segments_list:
                avg_confidence /= len(segments_list)
            
            return ASRResult(
                text=" ".join(full_text).strip(),
                confidence=float(avg_confidence) if avg_confidence > 0 else 0.85,
                language=info.language if hasattr(info, 'language') else self.language,
                tier="Tier1-Faster-Whisper",
                latency_ms=0,
                segments=segments_list
            )
        
        else:
            # Use openai-whisper
            result = self.tier1_model.transcribe(
                audio_path,
                language=self.language,
                fp16=self.use_gpu
            )
            
            return ASRResult(
                text=result["text"].strip(),
                confidence=0.90,  # Whisper doesn't provide confidence
                language=result.get("language", self.language),
                tier="Tier1-OpenAI-Whisper",
                latency_ms=0,
                segments=result.get("segments")
            )
    
    def _transcribe_vosk(self, audio_path: str) -> ASRResult:
        """Transcribe using Vosk"""
        # Open audio file
        wf = wave.open(audio_path, "rb")
        
        # Check format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
            logger.warning(f"Audio format may not be optimal for Vosk: {wf.getnchannels()} channels, {wf.getframerate()} Hz")
        
        # Create recognizer
        rec = KaldiRecognizer(self.tier2_model, wf.getframerate())
        rec.SetWords(True)
        
        # Process audio
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json_lib.loads(rec.Result())
                if 'text' in result:
                    results.append(result['text'])
        
        # Final result
        final_result = json_lib.loads(rec.FinalResult())
        if 'text' in final_result:
            results.append(final_result['text'])
        
        wf.close()
        
        full_text = " ".join(results).strip()
        
        return ASRResult(
            text=full_text,
            confidence=0.75,  # Vosk provides word-level confidence, approximate here
            language=self.language,
            tier="Tier2-Vosk",
            latency_ms=0
        )
    
    def transcribe_stream(self, audio_stream):
        """
        Transcribe streaming audio (primarily for Vosk).
        
        Args:
            audio_stream: Audio data stream
            
        Yields:
            Partial transcription results
        """
        if self.tier2_model is None:
            logger.error("Streaming requires Vosk (Tier 2)")
            return
        
        # Vosk streaming implementation would go here
        # For now, placeholder
        logger.warning("Streaming transcription not yet fully implemented")
        yield ASRResult(
            text="[Streaming not yet implemented]",
            confidence=0.0,
            language=self.language,
            tier="Tier2-Vosk-Stream",
            latency_ms=0
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get ASR status"""
        return {
            "tier1_whisper": self.tier1_model is not None,
            "tier1_type": getattr(self, 'tier1_type', None),
            "tier2_vosk": self.tier2_model is not None,
            "gpu_available": self.use_gpu,
            "device": self.device,
            "whisper_model_size": self.whisper_model_size
        }


# Example usage
if __name__ == "__main__":
    # Initialize ASR
    recognizer = ASRRecognizer(whisper_model_size="base")
    
    print("=" * 80)
    print("ASR RECOGNIZER - TESTING")
    print("=" * 80)
    print(f"\nStatus: {recognizer.get_status()}\n")
    
    print("Note: To test with real audio:")
    print("  1. Place a .wav file in the current directory")
    print("  2. Update the audio_path below")
    print("  3. Run: python recognizer.py")
    print("\nExample:")
    print("  result = recognizer.transcribe('test_audio.wav')")
    print("  print(result.text)")
    
    print("\n" + "=" * 80)

