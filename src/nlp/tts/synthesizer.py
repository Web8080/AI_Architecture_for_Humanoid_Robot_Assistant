"""
TTS (Text-to-Speech) with 3-Tier Fallback System
Tier 1: ElevenLabs (Best quality, cloud)
Tier 2: Coqui TTS (Good quality, local)
Tier 3: pyttsx3 (Fast, always works)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time
import torch

# Tier 1: ElevenLabs
try:
    from elevenlabs import generate, set_api_key, voices
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logging.warning("ElevenLabs not available. Install with: pip install elevenlabs")

# Tier 2: Coqui TTS
try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    logging.warning("Coqui TTS not available. Install with: pip install TTS")

# Tier 3: pyttsx3
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available. Install with: pip install pyttsx3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Represents TTS synthesis result"""
    audio_path: str
    tier: str
    model: str
    latency_ms: float
    duration_sec: float
    success: bool


class TTSSynthesizer:
    """
    Multi-tier TTS with automatic fallback.
    Tier 1: ElevenLabs (best quality, requires API key)
    Tier 2: Coqui TTS (good quality, local, GPU/CPU)
    Tier 3: pyttsx3 (instant, always works)
    """
    
    def __init__(
        self,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice: str = "Adam",  # Default voice
        coqui_model: str = "tts_models/en/ljspeech/vits",
        output_dir: str = "./audio_output",
        use_gpu: Optional[bool] = None,
        pyttsx3_rate: int = 175,
        pyttsx3_voice: Optional[str] = None
    ):
        """
        Initialize TTS synthesizer.
        
        Args:
            elevenlabs_api_key: ElevenLabs API key (from env if None)
            elevenlabs_voice: ElevenLabs voice name
            coqui_model: Coqui TTS model name
            output_dir: Directory for audio files
            use_gpu: Force GPU usage for Coqui (None = auto-detect)
            pyttsx3_rate: Speaking rate for pyttsx3
            pyttsx3_voice: Voice ID for pyttsx3
        """
        self.elevenlabs_voice = elevenlabs_voice
        self.coqui_model = coqui_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pyttsx3_rate = pyttsx3_rate
        self.pyttsx3_voice = pyttsx3_voice
        
        # Get ElevenLabs API key
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        
        # Auto-detect GPU
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
        
        logger.info(f"TTS using GPU: {self.use_gpu}")
        
        # Initialize tiers
        self.tier1_ready = False  # ElevenLabs
        self.tier2_model = None   # Coqui
        self.tier3_engine = None  # pyttsx3
        
        self._initialize_tier1()
        self._initialize_tier2()
        self._initialize_tier3()
    
    def _initialize_tier1(self):
        """Initialize Tier 1: ElevenLabs"""
        if not ELEVENLABS_AVAILABLE:
            logger.warning("Tier 1 (ElevenLabs) unavailable: elevenlabs package not installed")
            return
        
        if not self.elevenlabs_api_key:
            logger.warning("Tier 1 (ElevenLabs) unavailable: No API key provided")
            logger.info("Set ELEVENLABS_API_KEY environment variable to enable ElevenLabs")
            return
        
        try:
            set_api_key(self.elevenlabs_api_key)
            # Test by listing voices
            available_voices = voices()
            logger.info("✓ Tier 1 (ElevenLabs) initialized successfully")
            logger.info(f"  Available voices: {len(available_voices) if hasattr(available_voices, '__len__') else 'unknown'}")
            self.tier1_ready = True
        except Exception as e:
            logger.warning(f"ElevenLabs initialization failed: {e}")
            self.tier1_ready = False
    
    def _initialize_tier2(self):
        """Initialize Tier 2: Coqui TTS"""
        if not COQUI_AVAILABLE:
            logger.warning("Tier 2 (Coqui) unavailable: TTS package not installed")
            return
        
        try:
            logger.info(f"Loading Tier 2 (Coqui TTS): {self.coqui_model}")
            self.tier2_model = CoquiTTS(
                model_name=self.coqui_model,
                gpu=self.use_gpu
            )
            logger.info("✓ Tier 2 (Coqui TTS) initialized successfully")
        except Exception as e:
            logger.warning(f"Coqui TTS initialization failed: {e}")
            self.tier2_model = None
    
    def _initialize_tier3(self):
        """Initialize Tier 3: pyttsx3"""
        if not PYTTSX3_AVAILABLE:
            logger.warning("Tier 3 (pyttsx3) unavailable: pyttsx3 package not installed")
            return
        
        try:
            self.tier3_engine = pyttsx3.init()
            self.tier3_engine.setProperty('rate', self.pyttsx3_rate)
            
            if self.pyttsx3_voice:
                self.tier3_engine.setProperty('voice', self.pyttsx3_voice)
            
            logger.info("✓ Tier 3 (pyttsx3) initialized successfully")
        except Exception as e:
            logger.error(f"pyttsx3 initialization failed: {e}")
            self.tier3_engine = None
    
    def synthesize(self, text: str, output_filename: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech with automatic fallback.
        
        Args:
            text: Text to synthesize
            output_filename: Output audio filename (auto-generated if None)
            
        Returns:
            TTS result
        """
        start_time = time.time()
        
        if output_filename is None:
            timestamp = int(time.time() * 1000)
            output_filename = f"tts_output_{timestamp}.wav"
        
        output_path = self.output_dir / output_filename
        
        # Try Tier 1: ElevenLabs
        if self.tier1_ready:
            try:
                result = self._synthesize_elevenlabs(text, output_path)
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Synthesized using Tier 1: {text[:30]}...")
                return result
            except Exception as e:
                logger.warning(f"Tier 1 (ElevenLabs) failed: {e}. Falling back to Tier 2...")
        
        # Try Tier 2: Coqui TTS
        if self.tier2_model is not None:
            try:
                result = self._synthesize_coqui(text, output_path)
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Synthesized using Tier 2: {text[:30]}...")
                return result
            except Exception as e:
                logger.warning(f"Tier 2 (Coqui) failed: {e}. Falling back to Tier 3...")
        
        # Try Tier 3: pyttsx3
        if self.tier3_engine is not None:
            try:
                result = self._synthesize_pyttsx3(text, output_path)
                result.latency_ms = (time.time() - start_time) * 1000
                logger.debug(f"Synthesized using Tier 3: {text[:30]}...")
                return result
            except Exception as e:
                logger.error(f"All TTS tiers failed. Last error: {e}")
        
        # All failed
        return TTSResult(
            audio_path="",
            tier="Failed",
            model="none",
            latency_ms=(time.time() - start_time) * 1000,
            duration_sec=0.0,
            success=False
        )
    
    def _synthesize_elevenlabs(self, text: str, output_path: Path) -> TTSResult:
        """Synthesize using ElevenLabs"""
        audio = generate(
            text=text,
            voice=self.elevenlabs_voice,
            model="eleven_monolingual_v1"
        )
        
        # Save audio
        with open(output_path, 'wb') as f:
            f.write(audio)
        
        # Estimate duration (rough: ~150 words per minute)
        word_count = len(text.split())
        duration = (word_count / 150) * 60
        
        return TTSResult(
            audio_path=str(output_path),
            tier="Tier1-ElevenLabs",
            model=self.elevenlabs_voice,
            latency_ms=0,
            duration_sec=duration,
            success=True
        )
    
    def _synthesize_coqui(self, text: str, output_path: Path) -> TTSResult:
        """Synthesize using Coqui TTS"""
        self.tier2_model.tts_to_file(
            text=text,
            file_path=str(output_path)
        )
        
        # Get audio duration
        try:
            import wave
            with wave.open(str(output_path), 'r') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except Exception:
            word_count = len(text.split())
            duration = (word_count / 150) * 60
        
        return TTSResult(
            audio_path=str(output_path),
            tier="Tier2-Coqui",
            model=self.coqui_model,
            latency_ms=0,
            duration_sec=duration,
            success=True
        )
    
    def _synthesize_pyttsx3(self, text: str, output_path: Path) -> TTSResult:
        """Synthesize using pyttsx3"""
        self.tier3_engine.save_to_file(text, str(output_path))
        self.tier3_engine.runAndWait()
        
        # Estimate duration
        word_count = len(text.split())
        duration = (word_count / 150) * 60
        
        return TTSResult(
            audio_path=str(output_path),
            tier="Tier3-pyttsx3",
            model="pyttsx3-default",
            latency_ms=0,
            duration_sec=duration,
            success=True
        )
    
    def get_status(self) -> Dict[str, bool]:
        """Get TTS status"""
        return {
            "tier1_elevenlabs": self.tier1_ready,
            "tier2_coqui": self.tier2_model is not None,
            "tier3_pyttsx3": self.tier3_engine is not None,
            "gpu_available": self.use_gpu,
        }


# Example usage
if __name__ == "__main__":
    # Initialize TTS
    synthesizer = TTSSynthesizer()
    
    print("=" * 80)
    print("TTS SYNTHESIZER - TESTING")
    print("=" * 80)
    print(f"\nStatus: {synthesizer.get_status()}\n")
    
    # Test synthesis
    test_texts = [
        "Hello! I am your humanoid robot assistant.",
        "I can help you with navigation and object manipulation.",
    ]
    
    for text in test_texts:
        print(f"\nSynthesizing: {text}")
        result = synthesizer.synthesize(text)
        
        print(f"  Success: {result.success}")
        print(f"  Tier: {result.tier}")
        print(f"  Output: {result.audio_path}")
        print(f"  Latency: {result.latency_ms:.1f}ms")
        print(f"  Duration: {result.duration_sec:.1f}s")
    
    print("\n" + "=" * 80)

