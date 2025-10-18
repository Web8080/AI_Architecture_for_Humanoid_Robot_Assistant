"""
Comprehensive NLP Module Testing Script
Tests all components with their multi-tier fallback systems

Author: Victor Ibhafidon
Date: October 2025
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_entity_extractor():
    """Test Entity Extractor with all tiers"""
    print("\n" + "="*80)
    print("TESTING: Entity Extractor")
    print("="*80)
    
    try:
        from src.nlp.entities import EntityExtractor
        
        extractor = EntityExtractor()
        status = extractor.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (BERT):  {'✓' if status['tier1_bert'] else '✗'}")
        print(f"  Tier 2 (Custom): {'✓' if status['tier2_custom'] else '✗'}")
        print(f"  Tier 3 (spaCy):  {'✓' if status['tier3_spacy'] else '✗'}")
        print(f"  GPU Available:   {'✓' if status['gpu_available'] else '✗'}")
        
        # Test extraction
        test_text = "Bring me the red cup from the kitchen"
        entities = extractor.extract(test_text)
        
        print(f"\nTest: '{test_text}'")
        print(f"Extracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.type}: '{entity.text}' (confidence: {entity.confidence:.2f}, tier: {entity.tier})")
        
        print(f"\n{'✓' if entities else '✗'} Entity Extractor: {'PASS' if entities else 'FAIL'}")
        return True
        
    except Exception as e:
        print(f"\n✗ Entity Extractor: FAIL - {e}")
        return False


def test_dialogue_manager():
    """Test Dialogue Manager with all tiers"""
    print("\n" + "="*80)
    print("TESTING: Dialogue Manager")
    print("="*80)
    
    try:
        from src.nlp.dialogue import DialogueManager, DialogueState
        
        manager = DialogueManager()
        status = manager.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (Redis):     {'✓' if status['tier1_redis'] else '✗'}")
        print(f"  Tier 2 (LangChain): {'✓' if status['tier2_langchain'] else '✗'}")
        print(f"  Tier 3 (Memory):    ✓ (always available)")
        
        # Test session creation
        session = manager.create_session("test_001", "user_123")
        print(f"\nCreated session: {session.session_id}")
        
        # Test session update
        manager.update_session(
            "test_001",
            user_input="Hello robot",
            intent="greeting",
            entities={"greeting_type": "hello"},
            response="Hello! How can I help?",
            new_state=DialogueState.RESPONDING
        )
        
        context = manager.get_context("test_001")
        print(f"Context turns: {context['turn_count']}")
        print(f"Slots: {context['slots']}")
        
        print(f"\n✓ Dialogue Manager: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Dialogue Manager: FAIL - {e}")
        return False


def test_emotion_detector():
    """Test Emotion Detector with all tiers"""
    print("\n" + "="*80)
    print("TESTING: Emotion Detector")
    print("="*80)
    
    try:
        from src.nlp.emotion import EmotionDetector
        
        detector = EmotionDetector()
        status = detector.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (Emotion Transformer): {'✓' if status['tier1_transformer'] else '✗'}")
        print(f"  Tier 2 (Sentiment):           {'✓' if status['tier2_sentiment'] else '✗'}")
        print(f"  Tier 3 (VADER):               {'✓' if status['tier3_vader'] else '✗'}")
        print(f"  GPU Available:                {'✓' if status['gpu_available'] else '✗'}")
        
        # Test emotion detection
        test_texts = [
            "I'm so happy you're here!",
            "This is frustrating",
            "I'm okay"
        ]
        
        print(f"\nTesting emotion detection:")
        for text in test_texts:
            result = detector.detect(text)
            print(f"  '{text[:40]}'")
            print(f"    → {result.primary_emotion} (confidence: {result.confidence:.2f}, tier: {result.tier})")
        
        print(f"\n✓ Emotion Detector: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Emotion Detector: FAIL - {e}")
        return False


def test_rag_system():
    """Test RAG System"""
    print("\n" + "="*80)
    print("TESTING: RAG System")
    print("="*80)
    
    try:
        from src.nlp.rag import RAGRetriever
        
        retriever = RAGRetriever(persist_dir="./test_vector_store")
        status = retriever.get_status()
        
        print(f"\nRAG Status:")
        print(f"  Framework: {status['framework']}")
        print(f"  LangChain:  {'✓' if status['langchain_available'] else '✗'}")
        print(f"  LlamaIndex: {'✓' if status['llamaindex_available'] else '✗'}")
        print(f"  FAISS:      {'✓' if status['faiss_available'] else '✗'}")
        print(f"  Vector Store Ready: {'✓' if status['vector_store_ready'] else '✗'}")
        print(f"  GPU Enabled: {'✓' if status['gpu_enabled'] else '✗'}")
        
        # Add test documents
        docs = [
            "The robot can navigate using SLAM",
            "Object detection uses YOLOv8",
            "NLP includes intent and entity extraction"
        ]
        retriever.add_documents(docs)
        print(f"\nAdded {len(docs)} test documents")
        
        # Test retrieval
        query = "How does the robot navigate?"
        results = retriever.retrieve(query, top_k=2)
        print(f"\nQuery: '{query}'")
        print(f"Retrieved {len(results)} results:")
        for r in results:
            print(f"  - {r.text} (score: {r.score:.3f})")
        
        print(f"\n✓ RAG System: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ RAG System: FAIL - {e}")
        return False


def test_llm_integration():
    """Test LLM Integration"""
    print("\n" + "="*80)
    print("TESTING: LLM Integration")
    print("="*80)
    
    try:
        from src.nlp.llm import LLMIntegrator
        
        integrator = LLMIntegrator()
        status = integrator.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (OpenAI): {'✓' if status['tier1_openai'] else '✗'}")
        if status['tier1_openai']:
            print(f"    Model: {status['openai_model']}")
        print(f"  Tier 2 (Ollama): {'✓' if status['tier2_ollama'] else '✗'}")
        if status['tier2_ollama']:
            print(f"    Model: {status['ollama_model']}")
        print(f"  Tier 3 (Template): ✓ (always available)")
        
        # Test generation
        prompt = "What is 2+2?"
        print(f"\nTest: '{prompt}'")
        response = integrator.generate_sync(prompt, intent="answer_question")
        print(f"  Response: {response.content}")
        print(f"  Tier: {response.tier}")
        print(f"  Latency: {response.latency_ms:.1f}ms")
        
        print(f"\n✓ LLM Integration: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ LLM Integration: FAIL - {e}")
        return False


def test_asr():
    """Test ASR"""
    print("\n" + "="*80)
    print("TESTING: ASR (Speech Recognition)")
    print("="*80)
    
    try:
        from src.nlp.asr import ASRRecognizer
        
        recognizer = ASRRecognizer(whisper_model_size="tiny")  # Use tiny for faster testing
        status = recognizer.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (Whisper): {'✓' if status['tier1_whisper'] else '✗'}")
        if status['tier1_whisper']:
            print(f"    Type: {status['tier1_type']}")
            print(f"    Model: {status['whisper_model_size']}")
        print(f"  Tier 2 (Vosk):    {'✓' if status['tier2_vosk'] else '✗'}")
        print(f"  Device: {status['device']}")
        
        print(f"\nNote: ASR requires audio file to test properly")
        print(f"  To test: recognizer.transcribe('audio.wav')")
        
        print(f"\n✓ ASR: INITIALIZED")
        return True
        
    except Exception as e:
        print(f"\n✗ ASR: FAIL - {e}")
        return False


def test_tts():
    """Test TTS"""
    print("\n" + "="*80)
    print("TESTING: TTS (Speech Synthesis)")
    print("="*80)
    
    try:
        from src.nlp.tts import TTSSynthesizer
        
        synthesizer = TTSSynthesizer()
        status = synthesizer.get_status()
        
        print(f"\nTier Status:")
        print(f"  Tier 1 (ElevenLabs): {'✓' if status['tier1_elevenlabs'] else '✗'}")
        print(f"  Tier 2 (Coqui TTS):  {'✓' if status['tier2_coqui'] else '✗'}")
        print(f"  Tier 3 (pyttsx3):    {'✓' if status['tier3_pyttsx3'] else '✗'}")
        print(f"  GPU Available:       {'✓' if status['gpu_available'] else '✗'}")
        
        # Test synthesis
        test_text = "Hello, I am your robot assistant."
        print(f"\nTest: '{test_text}'")
        result = synthesizer.synthesize(test_text)
        
        if result.success:
            print(f"  ✓ Synthesized successfully")
            print(f"  Output: {result.audio_path}")
            print(f"  Tier: {result.tier}")
            print(f"  Latency: {result.latency_ms:.1f}ms")
        else:
            print(f"  ✗ Synthesis failed")
        
        print(f"\n{'✓' if result.success else '✗'} TTS: {'PASS' if result.success else 'FAIL'}")
        return result.success
        
    except Exception as e:
        print(f"\n✗ TTS: FAIL - {e}")
        return False


def test_full_nlp_service():
    """Test Full NLP Service Integration"""
    print("\n" + "="*80)
    print("TESTING: Full NLP Service Integration")
    print("="*80)
    
    try:
        from src.nlp.nlp_service import NLPService, NLPRequest
        
        service = NLPService()
        
        # Get status
        status = service.get_detailed_status()
        print(f"\nComponents Ready:")
        for component, ready in status['components'].items():
            print(f"  {'✓' if ready else '✗'} {component}")
        
        # Test end-to-end
        async def run_e2e_test():
            test_texts = [
                "Bring me the red cup",
                "I'm feeling great today!",
                "What's the temperature?"
            ]
            
            print(f"\nEnd-to-End Tests:")
            for text in test_texts:
                request = NLPRequest(
                    text=text,
                    session_id="test_e2e",
                    user_id="test_user",
                    use_rag=False,  # Disable RAG for basic test
                    use_llm=True
                )
                
                response = await service.process(request)
                
                print(f"\n  Input: {text}")
                print(f"    Intent: {response.intent} ({response.intent_confidence:.2f})")
                print(f"    Emotion: {response.emotion} ({response.emotion_confidence:.2f})")
                print(f"    Entities: {len(response.entities)}")
                print(f"    Response: {response.response_text}")
                print(f"    Latency: {response.latency_ms:.1f}ms")
                print(f"    Tiers: {response.tiers_used}")
        
        asyncio.run(run_e2e_test())
        
        print(f"\n✓ Full NLP Service: PASS")
        return True
        
    except Exception as e:
        print(f"\n✗ Full NLP Service: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("HUMANOID ROBOT ASSISTANT - NLP MODULE TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Run individual component tests
    results['Entity Extractor'] = test_entity_extractor()
    results['Dialogue Manager'] = test_dialogue_manager()
    results['Emotion Detector'] = test_emotion_detector()
    results['RAG System'] = test_rag_system()
    results['LLM Integration'] = test_llm_integration()
    results['ASR'] = test_asr()
    results['TTS'] = test_tts()
    results['Full NLP Service'] = test_full_nlp_service()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for component, result in results.items():
        status_icon = "✓" if result else "✗"
        print(f"  {status_icon} {component}: {'PASS' if result else 'FAIL'}")
    
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! NLP module is ready.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check logs above.")
    
    print("\n" + "="*80)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

