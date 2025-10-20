"""
Phase 3: Multimodal Fusion Integration Tests

PURPOSE:
    Comprehensive testing of Phase 3 multimodal fusion components including
    AI Agent Architecture, Visual Grounding, VQA, and Multimodal Fusion.

PIPELINE CONTEXT:
    
    Phase 3 Testing Flow:
    AI Agent → Multimodal Fusion → Visual Grounding → VQA → Integration
         ↓              ↓                ↓           ↓         ↓
    ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ ┌─────────┐
    │ Agent   │ │ Cross-Modal │ │ Referring   │ │ Visual  │ │ End-to- │
    │ States  │ │ Attention   │ │ Expression  │ │ Q&A     │ │ End     │
    │ Memory  │ │ Fusion      │ │ Grounding   │ │ System  │ │ Tests   │
    └─────────┘ └─────────────┘ └─────────────┘ └─────────┘ └─────────┘

WHY PHASE 3 TESTING MATTERS:
    Phase 3 bridges vision and language understanding:
    - Enables natural multimodal interaction
    - Supports visual grounding and VQA
    - Provides unified AI agent architecture
    - Enables cross-modal reasoning

HOW IT WORKS:
    1. Test AI Agent Architecture (perception → reasoning → planning → execution → learning)
    2. Test Multimodal Fusion (vision + language + audio)
    3. Test Visual Grounding (referring expressions)
    4. Test VQA (visual question answering)
    5. Test End-to-End Integration

INTEGRATION WITH EXISTING SYSTEM:
    - Builds on Phase 1 (NLP) and Phase 2 (Vision)
    - Adds multimodal fusion capabilities
    - Enables natural human-robot interaction
    - Provides foundation for Phase 4 (Task Planning)

RELATED FILES:
    - src/agents/ai_agent.py: AI Agent Architecture
    - src/agents/multimodal_fusion.py: Multimodal Fusion
    - src/multimodal/grounding/visual_grounding.py: Visual Grounding
    - src/multimodal/vqa/visual_qa.py: Visual Question Answering
    - configs/base/system_config.yaml: System configuration

USAGE:
    # Run Phase 3 tests
    python tests/integration/test_phase3_multimodal.py
    
    # Test specific component
    python tests/integration/test_phase3_multimodal.py --component ai_agent

Author: Victor Ibhafidon
Date: October 2025
Version: 1.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import logging
import numpy as np
import cv2
import argparse
import time
from typing import Dict, Any, List, Optional
import json

# Import Phase 3 components
from src.agents.ai_agent import AIAgent, AgentState
from src.agents.multimodal_fusion import MultimodalFusion
from src.multimodal.grounding.visual_grounding import VisualGrounding
from src.multimodal.vqa.visual_qa import VisualQuestionAnswering

# Import existing components
from src.nlp.nlp_service import NLPService
from src.vision.vision_service import VisionService

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Phase3Tester:
    """
    Comprehensive tester for Phase 3 multimodal fusion components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Phase 3 tester
        
        Args:
            config: Configuration for all Phase 3 components
        """
        self.config = config
        self.test_results = {}
        self.start_time = time.time()
        
        # Initialize components
        self._init_components()
        
        logger.info("Phase 3 Multimodal Fusion Tester initialized")
    
    def _init_components(self):
        """Initialize all Phase 3 components"""
        try:
            # AI Agent
            self.ai_agent = AIAgent(self.config.get('ai_agent', {}))
            logger.info("AI Agent initialized")
        except Exception as e:
            logger.error(f"AI Agent initialization failed: {e}")
            self.ai_agent = None
        
        try:
            # Multimodal Fusion
            self.multimodal_fusion = MultimodalFusion(self.config.get('multimodal_fusion', {}))
            logger.info("Multimodal Fusion initialized")
        except Exception as e:
            logger.error(f"Multimodal Fusion initialization failed: {e}")
            self.multimodal_fusion = None
        
        try:
            # Visual Grounding
            self.visual_grounding = VisualGrounding(self.config.get('visual_grounding', {}))
            logger.info("Visual Grounding initialized")
        except Exception as e:
            logger.error(f"Visual Grounding initialization failed: {e}")
            self.visual_grounding = None
        
        try:
            # Visual Question Answering
            self.vqa = VisualQuestionAnswering(self.config.get('vqa', {}))
            logger.info("Visual Question Answering initialized")
        except Exception as e:
            logger.error(f"VQA initialization failed: {e}")
            self.vqa = None
    
    async def test_ai_agent_architecture(self) -> Dict[str, Any]:
        """
        Test AI Agent Architecture (perception → reasoning → planning → execution → learning)
        """
        logger.info("Testing AI Agent Architecture...")
        
        test_results = {
            'component': 'AI Agent Architecture',
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
        if self.ai_agent is None:
            test_results['tests'].append({
                'test': 'AI Agent Initialization',
                'status': 'FAIL',
                'error': 'AI Agent not initialized'
            })
            test_results['failed'] += 1
            test_results['total'] += 1
            return test_results
        
        # Test 1: Agent State Management
        try:
            initial_state = self.ai_agent.state
            assert initial_state == AgentState.IDLE, f"Expected IDLE state, got {initial_state}"
            
            test_results['tests'].append({
                'test': 'Agent State Management',
                'status': 'PASS',
                'details': f'Initial state: {initial_state}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Agent State Management',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 2: Multi-modal Input Processing
        try:
            result = await self.ai_agent.process_input(
                text="Bring me a cup",
                image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                audio=None,
                sensors={'depth': 1.5}
            )
            
            assert 'response' in result, "Response not found in result"
            assert 'actions_taken' in result, "Actions not found in result"
            assert 'state' in result, "State not found in result"
            
            test_results['tests'].append({
                'test': 'Multi-modal Input Processing',
                'status': 'PASS',
                'details': f'Response generated, state: {result["state"]}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Multi-modal Input Processing',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 3: Autonomous Task Execution
        try:
            task_result = await self.ai_agent.execute_task("Find and bring me my phone")
            
            assert 'response' in task_result, "Response not found in task result"
            assert 'actions_taken' in task_result, "Actions not found in task result"
            
            test_results['tests'].append({
                'test': 'Autonomous Task Execution',
                'status': 'PASS',
                'details': 'Task execution completed'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Autonomous Task Execution',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 4: Agent Status and Memory
        try:
            status = self.ai_agent.get_agent_status()
            
            assert 'state' in status, "State not found in status"
            assert 'episodic_memory_size' in status, "Episodic memory size not found"
            assert 'semantic_memory_size' in status, "Semantic memory size not found"
            
            test_results['tests'].append({
                'test': 'Agent Status and Memory',
                'status': 'PASS',
                'details': f'Memory sizes: episodic={status["episodic_memory_size"]}, semantic={status["semantic_memory_size"]}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Agent Status and Memory',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        return test_results
    
    async def test_multimodal_fusion(self) -> Dict[str, Any]:
        """
        Test Multimodal Fusion (vision + language + audio)
        """
        logger.info("Testing Multimodal Fusion...")
        
        test_results = {
            'component': 'Multimodal Fusion',
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
        if self.multimodal_fusion is None:
            test_results['tests'].append({
                'test': 'Multimodal Fusion Initialization',
                'status': 'FAIL',
                'error': 'Multimodal Fusion not initialized'
            })
            test_results['failed'] += 1
            test_results['total'] += 1
            return test_results
        
        # Test 1: Vision Embedding Extraction
        try:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            vision_emb = await self.multimodal_fusion.extract_vision_embedding(test_image)
            
            assert vision_emb.modality == 'vision', f"Expected vision modality, got {vision_emb.modality}"
            assert len(vision_emb.embedding) > 0, "Vision embedding is empty"
            
            test_results['tests'].append({
                'test': 'Vision Embedding Extraction',
                'status': 'PASS',
                'details': f'Embedding shape: {vision_emb.embedding.shape}, confidence: {vision_emb.confidence:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Vision Embedding Extraction',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 2: Text Embedding Extraction
        try:
            text_emb = await self.multimodal_fusion.extract_text_embedding("What objects do you see?")
            
            assert text_emb.modality == 'text', f"Expected text modality, got {text_emb.modality}"
            assert len(text_emb.embedding) > 0, "Text embedding is empty"
            
            test_results['tests'].append({
                'test': 'Text Embedding Extraction',
                'status': 'PASS',
                'details': f'Embedding shape: {text_emb.embedding.shape}, confidence: {text_emb.confidence:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Text Embedding Extraction',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 3: Audio Embedding Extraction
        try:
            test_audio = np.random.randn(16000)
            audio_emb = await self.multimodal_fusion.extract_audio_embedding(test_audio)
            
            assert audio_emb.modality == 'audio', f"Expected audio modality, got {audio_emb.modality}"
            assert len(audio_emb.embedding) > 0, "Audio embedding is empty"
            
            test_results['tests'].append({
                'test': 'Audio Embedding Extraction',
                'status': 'PASS',
                'details': f'Embedding shape: {audio_emb.embedding.shape}, confidence: {audio_emb.confidence:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Audio Embedding Extraction',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 4: Modality Fusion
        try:
            fusion_result = await self.multimodal_fusion.fuse_modalities(
                vision_features=vision_emb,
                text_features=text_emb,
                audio_features=audio_emb
            )
            
            assert len(fusion_result.fused_embedding) > 0, "Fused embedding is empty"
            assert fusion_result.confidence >= 0.0, "Fusion confidence is negative"
            assert 'task_outputs' in fusion_result.__dict__, "Task outputs not found"
            
            test_results['tests'].append({
                'test': 'Modality Fusion',
                'status': 'PASS',
                'details': f'Fused embedding shape: {fusion_result.fused_embedding.shape}, confidence: {fusion_result.confidence:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Modality Fusion',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 5: Visual Question Answering
        try:
            vqa_result = await self.multimodal_fusion.visual_question_answering(
                image=test_image,
                question="What do you see?"
            )
            
            assert 'answer' in vqa_result, "Answer not found in VQA result"
            assert 'confidence' in vqa_result, "Confidence not found in VQA result"
            
            test_results['tests'].append({
                'test': 'Visual Question Answering',
                'status': 'PASS',
                'details': f'Answer: "{vqa_result["answer"]}", confidence: {vqa_result["confidence"]:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Visual Question Answering',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        return test_results
    
    async def test_visual_grounding(self) -> Dict[str, Any]:
        """
        Test Visual Grounding (referring expressions)
        """
        logger.info("Testing Visual Grounding...")
        
        test_results = {
            'component': 'Visual Grounding',
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
        if self.visual_grounding is None:
            test_results['tests'].append({
                'test': 'Visual Grounding Initialization',
                'status': 'FAIL',
                'error': 'Visual Grounding not initialized'
            })
            test_results['failed'] += 1
            test_results['total'] += 1
            return test_results
        
        # Test 1: Expression Parsing
        try:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            expression = "the red cup on the left"
            
            result = await self.visual_grounding.ground_expression(test_image, expression)
            
            assert hasattr(result, 'grounded'), "Grounded attribute not found"
            assert hasattr(result, 'confidence'), "Confidence attribute not found"
            assert hasattr(result, 'reasoning'), "Reasoning attribute not found"
            
            test_results['tests'].append({
                'test': 'Expression Parsing and Grounding',
                'status': 'PASS',
                'details': f'Grounded: {result.grounded}, confidence: {result.confidence:.2f}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Expression Parsing and Grounding',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 2: Batch Grounding
        try:
            expressions = [
                "the red cup",
                "the object on the left",
                "that small bottle",
                "the round object"
            ]
            
            batch_results = await self.visual_grounding.batch_ground_expressions(test_image, expressions)
            
            assert len(batch_results) == len(expressions), f"Expected {len(expressions)} results, got {len(batch_results)}"
            
            for i, result in enumerate(batch_results):
                assert hasattr(result, 'grounded'), f"Result {i} missing grounded attribute"
                assert hasattr(result, 'confidence'), f"Result {i} missing confidence attribute"
            
            test_results['tests'].append({
                'test': 'Batch Grounding',
                'status': 'PASS',
                'details': f'Processed {len(expressions)} expressions successfully'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Batch Grounding',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 3: Grounding Statistics
        try:
            stats = self.visual_grounding.get_grounding_statistics()
            
            assert 'confidence_threshold' in stats, "Confidence threshold not found in stats"
            assert 'supported_colors' in stats, "Supported colors not found in stats"
            assert 'supported_sizes' in stats, "Supported sizes not found in stats"
            
            test_results['tests'].append({
                'test': 'Grounding Statistics',
                'status': 'PASS',
                'details': f'Supported colors: {len(stats["supported_colors"])}, sizes: {len(stats["supported_sizes"])}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Grounding Statistics',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        return test_results
    
    async def test_visual_question_answering(self) -> Dict[str, Any]:
        """
        Test Visual Question Answering
        """
        logger.info("Testing Visual Question Answering...")
        
        test_results = {
            'component': 'Visual Question Answering',
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
        if self.vqa is None:
            test_results['tests'].append({
                'test': 'VQA Initialization',
                'status': 'FAIL',
                'error': 'VQA not initialized'
            })
            test_results['failed'] += 1
            test_results['total'] += 1
            return test_results
        
        # Test 1: Single Question Answering
        try:
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            question = "What objects do you see?"
            
            result = await self.vqa.answer_question(test_image, question)
            
            assert hasattr(result, 'answer'), "Answer attribute not found"
            assert hasattr(result, 'confidence'), "Confidence attribute not found"
            assert hasattr(result, 'question_type'), "Question type attribute not found"
            assert hasattr(result, 'reasoning'), "Reasoning attribute not found"
            
            test_results['tests'].append({
                'test': 'Single Question Answering',
                'status': 'PASS',
                'details': f'Answer: "{result.answer}", confidence: {result.confidence:.2f}, type: {result.question_type.value}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Single Question Answering',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 2: Batch Question Answering
        try:
            questions = [
                "What objects do you see?",
                "How many objects are there?",
                "What color is the cup?",
                "Is there a person in the image?",
                "Describe what you see"
            ]
            
            batch_results = await self.vqa.batch_answer_questions(test_image, questions)
            
            assert len(batch_results) == len(questions), f"Expected {len(questions)} results, got {len(batch_results)}"
            
            for i, result in enumerate(batch_results):
                assert hasattr(result, 'answer'), f"Result {i} missing answer attribute"
                assert hasattr(result, 'confidence'), f"Result {i} missing confidence attribute"
            
            test_results['tests'].append({
                'test': 'Batch Question Answering',
                'status': 'PASS',
                'details': f'Processed {len(questions)} questions successfully'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Batch Question Answering',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 3: VQA Statistics
        try:
            stats = self.vqa.get_vqa_statistics()
            
            assert 'vocab_size' in stats, "Vocab size not found in stats"
            assert 'max_answer_length' in stats, "Max answer length not found in stats"
            assert 'supported_question_types' in stats, "Supported question types not found in stats"
            
            test_results['tests'].append({
                'test': 'VQA Statistics',
                'status': 'PASS',
                'details': f'Vocab size: {stats["vocab_size"]}, question types: {len(stats["supported_question_types"])}'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'VQA Statistics',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        return test_results
    
    async def test_end_to_end_integration(self) -> Dict[str, Any]:
        """
        Test end-to-end integration of all Phase 3 components
        """
        logger.info("Testing End-to-End Integration...")
        
        test_results = {
            'component': 'End-to-End Integration',
            'tests': [],
            'passed': 0,
            'failed': 0,
            'total': 0
        }
        
        # Test 1: Complete AI Agent Workflow
        try:
            if self.ai_agent is not None:
                # Simulate complete workflow
                result = await self.ai_agent.process_input(
                    text="What objects do you see in this image?",
                    image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    audio=None,
                    sensors={'depth': 1.5}
                )
                
                assert 'response' in result, "Response not found in end-to-end result"
                assert 'actions_taken' in result, "Actions not found in end-to-end result"
                assert 'learning_updates' in result, "Learning updates not found in end-to-end result"
                
                test_results['tests'].append({
                    'test': 'Complete AI Agent Workflow',
                    'status': 'PASS',
                    'details': 'End-to-end workflow completed successfully'
                })
                test_results['passed'] += 1
            else:
                test_results['tests'].append({
                    'test': 'Complete AI Agent Workflow',
                    'status': 'SKIP',
                    'details': 'AI Agent not available'
                })
        except Exception as e:
            test_results['tests'].append({
                'test': 'Complete AI Agent Workflow',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 2: Multimodal Fusion Integration
        try:
            if self.multimodal_fusion is not None and self.vqa is not None:
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Test VQA through multimodal fusion
                vqa_result = await self.multimodal_fusion.visual_question_answering(
                    image=test_image,
                    question="What do you see?"
                )
                
                assert 'answer' in vqa_result, "Answer not found in multimodal VQA result"
                
                test_results['tests'].append({
                    'test': 'Multimodal Fusion Integration',
                    'status': 'PASS',
                    'details': f'Multimodal VQA answer: "{vqa_result["answer"]}"'
                })
                test_results['passed'] += 1
            else:
                test_results['tests'].append({
                    'test': 'Multimodal Fusion Integration',
                    'status': 'SKIP',
                    'details': 'Required components not available'
                })
        except Exception as e:
            test_results['tests'].append({
                'test': 'Multimodal Fusion Integration',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        # Test 3: Component Interoperability
        try:
            component_count = 0
            available_components = []
            
            if self.ai_agent is not None:
                component_count += 1
                available_components.append('AI Agent')
            
            if self.multimodal_fusion is not None:
                component_count += 1
                available_components.append('Multimodal Fusion')
            
            if self.visual_grounding is not None:
                component_count += 1
                available_components.append('Visual Grounding')
            
            if self.vqa is not None:
                component_count += 1
                available_components.append('VQA')
            
            assert component_count > 0, "No Phase 3 components available"
            
            test_results['tests'].append({
                'test': 'Component Interoperability',
                'status': 'PASS',
                'details': f'Available components: {", ".join(available_components)} ({component_count}/4)'
            })
            test_results['passed'] += 1
        except Exception as e:
            test_results['tests'].append({
                'test': 'Component Interoperability',
                'status': 'FAIL',
                'error': str(e)
            })
            test_results['failed'] += 1
        test_results['total'] += 1
        
        return test_results
    
    async def run_all_tests(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Run all Phase 3 tests or specific component tests
        
        Args:
            component: Specific component to test (optional)
        
        Returns:
            Complete test results
        """
        logger.info("Starting Phase 3 Multimodal Fusion Tests...")
        
        all_results = {
            'phase': 'Phase 3: Multimodal Fusion',
            'timestamp': time.time(),
            'components': {},
            'summary': {
                'total_tests': 0,
                'total_passed': 0,
                'total_failed': 0,
                'total_skipped': 0
            }
        }
        
        # Define test functions
        test_functions = {
            'ai_agent': self.test_ai_agent_architecture,
            'multimodal_fusion': self.test_multimodal_fusion,
            'visual_grounding': self.test_visual_grounding,
            'vqa': self.test_visual_question_answering,
            'integration': self.test_end_to_end_integration
        }
        
        # Run tests
        if component and component in test_functions:
            # Run specific component
            logger.info(f"Running tests for component: {component}")
            result = await test_functions[component]()
            all_results['components'][component] = result
        else:
            # Run all tests
            for comp_name, test_func in test_functions.items():
                logger.info(f"Running tests for component: {comp_name}")
                result = await test_func()
                all_results['components'][comp_name] = result
        
        # Calculate summary
        for comp_result in all_results['components'].values():
            all_results['summary']['total_tests'] += comp_result['total']
            all_results['summary']['total_passed'] += comp_result['passed']
            all_results['summary']['total_failed'] += comp_result['failed']
            all_results['summary']['total_skipped'] += comp_result.get('skipped', 0)
        
        # Calculate success rate
        if all_results['summary']['total_tests'] > 0:
            success_rate = (all_results['summary']['total_passed'] / all_results['summary']['total_tests']) * 100
            all_results['summary']['success_rate'] = success_rate
        else:
            all_results['summary']['success_rate'] = 0.0
        
        # Add timing information
        all_results['execution_time'] = time.time() - self.start_time
        
        return all_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print(f"PHASE 3: MULTIMODAL FUSION TEST RESULTS")
        print("="*80)
        
        print(f"\nTest Summary:")
        print(f"  Total Tests: {results['summary']['total_tests']}")
        print(f"  Passed: {results['summary']['total_passed']}")
        print(f"  Failed: {results['summary']['total_failed']}")
        print(f"  Skipped: {results['summary']['total_skipped']}")
        print(f"  Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"  Execution Time: {results['execution_time']:.2f}s")
        
        print(f"\nComponent Results:")
        for comp_name, comp_result in results['components'].items():
            print(f"\n  {comp_result['component']}:")
            print(f"    Tests: {comp_result['passed']}/{comp_result['total']} passed")
            
            for test in comp_result['tests']:
                status_symbol = "✓" if test['status'] == 'PASS' else "✗" if test['status'] == 'FAIL' else "○"
                print(f"    {status_symbol} {test['test']}: {test['status']}")
                if 'details' in test:
                    print(f"      {test['details']}")
                if 'error' in test:
                    print(f"      Error: {test['error']}")
        
        print("\n" + "="*80)
        
        # Overall status
        if results['summary']['success_rate'] >= 80:
            print("PHASE 3 STATUS: READY FOR PRODUCTION")
        elif results['summary']['success_rate'] >= 60:
            print("PHASE 3 STATUS: NEEDS IMPROVEMENT")
        else:
            print("PHASE 3 STATUS: REQUIRES FIXES")
        
        print("="*80)


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Phase 3 Multimodal Fusion Tests')
    parser.add_argument('--component', type=str, help='Test specific component')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'ai_agent': {
            'nlp': {
                'intent_classifier': {'enabled': True},
                'entity_extractor': {'enabled': True},
                'emotion_detector': {'enabled': True},
                'dialogue': {'enabled': True},
                'rag': {'enabled': True},
                'llm': {'enabled': True},
                'asr': {'enabled': True},
                'tts': {'enabled': True}
            },
            'vision': {
                'object_detection': {'enabled': True},
                'segmentation': {'enabled': True},
                'depth_estimation': {'enabled': True},
                'pose_estimation': {'enabled': True},
                'face_recognition': {'enabled': True},
                'scene_understanding': {'enabled': True}
            }
        },
        'multimodal_fusion': {
            'vision_dim': 512,
            'text_dim': 768,
            'audio_dim': 256,
            'hidden_dim': 512,
            'output_dim': 256,
            'num_attention_layers': 3
        },
        'visual_grounding': {
            'object_detection': {
                'tier1_enabled': True,
                'tier2_enabled': True,
                'tier3_enabled': True
            },
            'entity_extraction': {
                'tier1_enabled': True,
                'tier2_enabled': True,
                'tier3_enabled': True
            },
            'confidence_threshold': 0.3,
            'max_alternatives': 3
        },
        'vqa': {
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
    }
    
    # Load custom config if provided
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, 'r') as f:
            custom_config = yaml.safe_load(f)
            config.update(custom_config)
    
    # Initialize tester
    tester = Phase3Tester(config)
    
    # Run tests
    results = await tester.run_all_tests(component=args.component)
    
    # Print results
    tester.print_results(results)
    
    # Save results
    results_file = Path('test_results') / f'phase3_test_results_{int(time.time())}.json'
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code based on success rate
    if results['summary']['success_rate'] >= 80:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
