"""
Unified NLP Service
Integrates all NLP components with multi-tier fallback system:
- Intent Classification
- Entity Extraction
- Dialogue Management
- Emotion Detection
- RAG (Retrieval-Augmented Generation)
- LLM Integration
- ASR (Speech Recognition)
- TTS (Speech Synthesis)

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

# Import our NLP components
try:
    from .intent.classifier import IntentClassifier
    INTENT_AVAILABLE = True
except ImportError:
    INTENT_AVAILABLE = False
    logging.warning("Intent classifier not available")

try:
    from .entities.extractor import EntityExtractor
    ENTITY_AVAILABLE = True
except ImportError:
    ENTITY_AVAILABLE = False
    logging.warning("Entity extractor not available")

try:
    from .dialogue.manager import DialogueManager, DialogueState
    DIALOGUE_AVAILABLE = True
except ImportError:
    DIALOGUE_AVAILABLE = False
    logging.warning("Dialogue manager not available")

try:
    from .emotion.detector import EmotionDetector
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    logging.warning("Emotion detector not available")

try:
    from .rag.retriever import RAGRetriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG retriever not available")

try:
    from .llm.integrator import LLMIntegrator
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM integrator not available")

try:
    from .asr.recognizer import ASRRecognizer
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    logging.warning("ASR recognizer not available")

try:
    from .tts.synthesizer import TTSSynthesizer
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS synthesizer not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NLPRequest:
    """Request to NLP service"""
    text: Optional[str] = None
    audio_path: Optional[str] = None
    session_id: str = "default"
    user_id: str = "default_user"
    context: Optional[Dict[str, Any]] = None
    use_rag: bool = True
    use_llm: bool = True


@dataclass
class NLPResponse:
    """Response from NLP service"""
    # Core NLP results
    intent: Optional[str]
    intent_confidence: float
    entities: List[Dict[str, Any]]
    emotion: Optional[str]
    emotion_confidence: float
    
    # Generated response
    response_text: str
    response_audio_path: Optional[str]
    
    # Context and metadata
    session_id: str
    dialogue_state: str
    clarifications_needed: List[str]
    slots_filled: Dict[str, Any]
    
    # Performance metrics
    latency_ms: float
    tiers_used: Dict[str, str]
    
    # RAG info (if used)
    rag_context: Optional[str] = None
    rag_sources: Optional[List[str]] = None


class NLPService:
    """
    Unified NLP Service integrating all components.
    Provides end-to-end natural language understanding and generation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize NLP service with all components.
        
        Args:
            config: Configuration dictionary (from YAML)
        """
        self.config = config or {}
        
        # Initialize components
        logger.info("Initializing NLP Service components...")
        
        # Intent Classifier
        self.intent_classifier = IntentClassifier() if INTENT_AVAILABLE else None
        
        # Entity Extractor
        self.entity_extractor = EntityExtractor() if ENTITY_AVAILABLE else None
        
        # Dialogue Manager
        dialogue_config = self.config.get('nlp', {}).get('dialogue', {})
        self.dialogue_manager = DialogueManager(
            redis_host=dialogue_config.get('redis_host', 'localhost'),
            redis_port=dialogue_config.get('redis_port', 6379),
            session_ttl=dialogue_config.get('session_ttl_minutes', 15) * 60,
            context_window=dialogue_config.get('context_window_turns', 10)
        ) if DIALOGUE_AVAILABLE else None
        
        # Emotion Detector
        self.emotion_detector = EmotionDetector() if EMOTION_AVAILABLE else None
        
        # RAG Retriever
        rag_config = self.config.get('nlp', {}).get('rag', {})
        self.rag_retriever = RAGRetriever(
            vector_store_type=rag_config.get('vector_store_type', 'faiss'),
            persist_dir=rag_config.get('faiss', {}).get('persist_dir', './data/vector_store/faiss')
        ) if RAG_AVAILABLE else None
        
        # LLM Integrator
        llm_config = self.config.get('nlp', {}).get('llm', {})
        self.llm_integrator = LLMIntegrator(
            openai_model=llm_config.get('tier1_openai', {}).get('model', 'gpt-4o-mini'),
            ollama_model=llm_config.get('tier2_ollama', {}).get('model', 'llama3.2:3b')
        ) if LLM_AVAILABLE else None
        
        # ASR Recognizer
        asr_config = self.config.get('nlp', {}).get('asr', {})
        self.asr_recognizer = ASRRecognizer(
            whisper_model_size=asr_config.get('tier1_whisper', {}).get('model_size', 'base')
        ) if ASR_AVAILABLE else None
        
        # TTS Synthesizer
        self.tts_synthesizer = TTSSynthesizer() if TTS_AVAILABLE else None
        
        logger.info("✓ NLP Service initialized")
        logger.info(f"  Components ready: {self.get_component_status()}")
    
    async def process(self, request: NLPRequest) -> NLPResponse:
        """
        Process NLP request end-to-end.
        
        Args:
            request: NLP request
            
        Returns:
            NLP response
        """
        start_time = time.time()
        tiers_used = {}
        
        # 1. Handle audio input (ASR)
        if request.audio_path and self.asr_recognizer:
            try:
                asr_result = self.asr_recognizer.transcribe(request.audio_path)
                request.text = asr_result.text
                tiers_used['asr'] = asr_result.tier
            except Exception as e:
                logger.error(f"ASR failed: {e}")
        
        if not request.text:
            return self._error_response(request.session_id, "No input text provided")
        
        # 2. Detect intent
        intent = None
        intent_confidence = 0.0
        if self.intent_classifier:
            try:
                intent, intent_confidence = self.intent_classifier.predict(request.text)
                tiers_used['intent'] = 'IntentClassifier'
            except Exception as e:
                logger.error(f"Intent classification failed: {e}")
        
        # 3. Extract entities
        entities = []
        if self.entity_extractor:
            try:
                entity_results = self.entity_extractor.extract(request.text)
                entities = [asdict(e) for e in entity_results]
                if entity_results:
                    tiers_used['entities'] = entity_results[0].tier
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
        
        # 4. Detect emotion
        emotion = None
        emotion_confidence = 0.0
        if self.emotion_detector:
            try:
                emotion_result = self.emotion_detector.detect(request.text)
                emotion = emotion_result.primary_emotion
                emotion_confidence = emotion_result.confidence
                tiers_used['emotion'] = emotion_result.tier
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
        
        # 5. Update dialogue state
        dialogue_state = DialogueState.IDLE.value
        slots_filled = {}
        clarifications_needed = []
        
        if self.dialogue_manager:
            try:
                # Get or create session
                session = self.dialogue_manager.get_session(request.session_id)
                if session is None:
                    session = self.dialogue_manager.create_session(
                        request.session_id,
                        request.user_id,
                        request.context
                    )
                
                # Convert entities to dict for slots
                entity_dict = {e['type'].lower(): e['text'] for e in entities}
                
                # Update session
                session = self.dialogue_manager.update_session(
                    request.session_id,
                    user_input=request.text,
                    intent=intent,
                    entities=entity_dict,
                    new_state=DialogueState.UNDERSTANDING
                )
                
                dialogue_state = session.state
                slots_filled = session.slots
                tiers_used['dialogue'] = 'Tier1-StateMachine'
                
            except Exception as e:
                logger.error(f"Dialogue management failed: {e}")
        
        # 6. Retrieve context from RAG (if enabled)
        rag_context = None
        rag_sources = []
        if request.use_rag and self.rag_retriever:
            try:
                context_text, retrieval_results = self.rag_retriever.query_with_context(
                    request.text,
                    top_k=3
                )
                rag_context = context_text
                rag_sources = [r.metadata.get('source', 'unknown') for r in retrieval_results]
                tiers_used['rag'] = retrieval_results[0].source if retrieval_results else 'None'
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # 7. Generate response with LLM (if enabled)
        response_text = "I understand."
        if request.use_llm and self.llm_integrator:
            try:
                system_prompt = "You are a helpful humanoid robot assistant."
                llm_result = self.llm_integrator.generate_sync(
                    prompt=request.text,
                    system_prompt=system_prompt,
                    context=rag_context,
                    intent=intent
                )
                response_text = llm_result.content
                tiers_used['llm'] = llm_result.tier
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
        
        # 8. Synthesize speech (TTS)
        response_audio_path = None
        if self.tts_synthesizer:
            try:
                tts_result = self.tts_synthesizer.synthesize(response_text)
                if tts_result.success:
                    response_audio_path = tts_result.audio_path
                    tiers_used['tts'] = tts_result.tier
            except Exception as e:
                logger.warning(f"TTS synthesis failed: {e}")
        
        # 9. Update dialogue with response
        if self.dialogue_manager:
            try:
                self.dialogue_manager.update_session(
                    request.session_id,
                    user_input=request.text,
                    intent=intent,
                    entities=entity_dict,
                    response=response_text,
                    new_state=DialogueState.RESPONDING
                )
            except Exception as e:
                logger.warning(f"Dialogue update failed: {e}")
        
        # Calculate total latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Build response
        return NLPResponse(
            intent=intent,
            intent_confidence=intent_confidence,
            entities=entities,
            emotion=emotion,
            emotion_confidence=emotion_confidence,
            response_text=response_text,
            response_audio_path=response_audio_path,
            session_id=request.session_id,
            dialogue_state=dialogue_state,
            clarifications_needed=clarifications_needed,
            slots_filled=slots_filled,
            latency_ms=latency_ms,
            tiers_used=tiers_used,
            rag_context=rag_context,
            rag_sources=rag_sources
        )
    
    def _error_response(self, session_id: str, error_message: str) -> NLPResponse:
        """Generate error response"""
        return NLPResponse(
            intent="error",
            intent_confidence=0.0,
            entities=[],
            emotion="neutral",
            emotion_confidence=0.0,
            response_text=f"Error: {error_message}",
            response_audio_path=None,
            session_id=session_id,
            dialogue_state=DialogueState.ERROR.value,
            clarifications_needed=[],
            slots_filled={},
            latency_ms=0.0,
            tiers_used={}
        )
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all NLP components"""
        return {
            "intent_classifier": self.intent_classifier is not None,
            "entity_extractor": self.entity_extractor is not None,
            "dialogue_manager": self.dialogue_manager is not None,
            "emotion_detector": self.emotion_detector is not None,
            "rag_retriever": self.rag_retriever is not None,
            "llm_integrator": self.llm_integrator is not None,
            "asr_recognizer": self.asr_recognizer is not None,
            "tts_synthesizer": self.tts_synthesizer is not None,
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of all components and their tiers"""
        status = {"components": self.get_component_status()}
        
        # Get tier status for each component
        if self.entity_extractor:
            status['entity_extractor_tiers'] = self.entity_extractor.get_status()
        
        if self.dialogue_manager:
            status['dialogue_manager_tiers'] = self.dialogue_manager.get_status()
        
        if self.emotion_detector:
            status['emotion_detector_tiers'] = self.emotion_detector.get_status()
        
        if self.rag_retriever:
            status['rag_retriever_status'] = self.rag_retriever.get_status()
        
        if self.llm_integrator:
            status['llm_integrator_tiers'] = self.llm_integrator.get_status()
        
        if self.asr_recognizer:
            status['asr_recognizer_tiers'] = self.asr_recognizer.get_status()
        
        if self.tts_synthesizer:
            status['tts_synthesizer_tiers'] = self.tts_synthesizer.get_status()
        
        return status


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize NLP service
    service = NLPService()
    
    print("=" * 80)
    print("NLP SERVICE - COMPREHENSIVE TESTING")
    print("=" * 80)
    
    # Get status
    status = service.get_detailed_status()
    print("\nComponent Status:")
    for component, ready in status['components'].items():
        status_icon = "✓" if ready else "✗"
        print(f"  {status_icon} {component}: {ready}")
    
    print("\n" + "=" * 80)
    print("TESTING TEXT PROCESSING")
    print("=" * 80)
    
    # Test requests
    test_requests = [
        "Bring me the red cup from the kitchen",
        "I'm feeling really happy today!",
        "What's the weather like?",
        "Stop moving immediately",
    ]
    
    async def run_tests():
        for i, text in enumerate(test_requests, 1):
            print(f"\n[Test {i}] Input: {text}")
            
            request = NLPRequest(
                text=text,
                session_id=f"test_session_{i}",
                user_id="test_user",
                use_rag=False,  # Disable for basic test
                use_llm=True
            )
            
            response = await service.process(request)
            
            print(f"  Intent: {response.intent} (confidence: {response.intent_confidence:.2f})")
            print(f"  Entities: {response.entities}")
            print(f"  Emotion: {response.emotion} (confidence: {response.emotion_confidence:.2f})")
            print(f"  Response: {response.response_text}")
            print(f"  Latency: {response.latency_ms:.1f}ms")
            print(f"  Tiers used: {response.tiers_used}")
    
    # Run async tests
    asyncio.run(run_tests())
    
    print("\n" + "=" * 80)

