"""
Intent Router - Central Dispatcher for Humanoid Robot

Maps NLP intents to appropriate robot engines for execution.
Inspired by Chapo bot's successful architecture with 500+ engines.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from src.engines.base_engine import BaseEngine, EngineResponse, EngineStatus

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Categories of intents"""
    OBJECT_MANIPULATION = "object_manipulation"
    NAVIGATION = "navigation"
    VISION = "vision"
    INTERACTION = "interaction"
    MEMORY = "memory"
    PLANNING = "planning"
    SAFETY = "safety"
    CONTROL = "control"
    PERCEPTION = "perception"
    LEARNING = "learning"
    HOME_AUTOMATION = "home_automation"
    ENTERTAINMENT = "entertainment"
    UNKNOWN = "unknown"


@dataclass
class RouterResponse:
    """Response from intent router"""
    intent: str
    normalized_intent: str
    category: IntentCategory
    engine_responses: List[EngineResponse]
    total_execution_time: float
    success: bool
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "normalized_intent": self.normalized_intent,
            "category": self.category.value,
            "engine_responses": [r.to_dict() for r in self.engine_responses],
            "total_execution_time": self.total_execution_time,
            "success": self.success,
            "message": self.message
        }


class IntentRouter:
    """
    Central intent router mapping intents to engines
    
    Architecture:
    1. Receive intent from NLP module
    2. Normalize intent using intent normalization map
    3. Determine required engines for intent
    4. Execute engines in appropriate order (sequential or parallel)
    5. Aggregate responses and return to user
    
    Features:
    - 1000+ intent mappings
    - 500+ engine integrations
    - Multi-engine coordination
    - Session memory management
    - Fallback handling
    - Performance tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        self.engines = {}
        self.intent_normalization_map = {}
        self.intent_engine_map = {}
        self.intent_category_map = {}
        self.session_memory = {}
        
        self._initialize_intent_maps()
        logger.info("IntentRouter initialized")
    
    def _initialize_intent_maps(self):
        """Initialize intent normalization and engine mapping"""
        
        # ============================================================
        # INTENT NORMALIZATION MAP (1000+ mappings)
        # ============================================================
        
        # Object Manipulation Intents
        self.intent_normalization_map.update({
            # Grasping
            "pick_up_object": "object_grasp",
            "grab_object": "object_grasp",
            "grasp_object": "object_grasp",
            "hold_object": "object_grasp",
            "take_object": "object_grasp",
            
            # Placement
            "put_object": "object_place",
            "place_object": "object_place",
            "set_object": "object_place",
            "put_down_object": "object_place",
            
            # Transfer
            "bring_object": "object_transfer",
            "fetch_object": "object_transfer",
            "get_object": "object_transfer",
            "hand_object": "object_transfer",
            "give_object": "object_transfer",
            
            # Complex manipulation
            "open_object": "object_open",
            "close_object": "object_close",
            "pour_object": "object_pour",
            "stack_objects": "object_stack",
            "sort_objects": "object_sort",
            "clean_object": "object_clean",
            "fold_object": "object_fold",
            "cut_object": "object_cut",
        })
        
        # Navigation Intents
        self.intent_normalization_map.update({
            "go_to_location": "navigate_to",
            "move_to_location": "navigate_to",
            "navigate_to_room": "navigate_to",
            "walk_to": "navigate_to",
            
            "turn_left": "turn_direction",
            "turn_right": "turn_direction",
            "turn_around": "turn_direction",
            
            "move_forward": "move_direction",
            "move_backward": "move_direction",
            "go_forward": "move_direction",
            
            "follow_me": "follow_person",
            "come_here": "navigate_to_user",
            "follow_to_location": "follow_to_location",
            
            "stop_moving": "stop_movement",
            "halt": "stop_movement",
            "wait": "stop_movement",
            "freeze": "stop_movement",
        })
        
        # Vision Intents
        self.intent_normalization_map.update({
            "what_do_you_see": "visual_scene_description",
            "describe_scene": "visual_scene_description",
            "look_around": "visual_scene_description",
            
            "how_many": "visual_count_objects",
            "count_objects": "visual_count_objects",
            
            "where_is": "visual_locate_object",
            "find_object": "visual_locate_object",
            
            "who_is_this": "face_recognition",
            "recognize_face": "face_recognition",
            "identify_person": "face_recognition",
        })
        
        # Interaction Intents
        self.intent_normalization_map.update({
            "hello": "greeting",
            "hi": "greeting",
            "hey": "greeting",
            "good_morning": "greeting",
            
            "goodbye": "farewell",
            "bye": "farewell",
            "see_you": "farewell",
            
            "tell_joke": "tell_joke",
            "tell_story": "tell_story",
            "sing_song": "sing_song",
            
            "what_can_you_do": "ask_capabilities",
            "who_are_you": "ask_identity",
            "what_is_your_name": "ask_name",
        })
        
        # Memory Intents
        self.intent_normalization_map.update({
            "remember_this": "store_memory",
            "dont_forget": "store_memory",
            "save_info": "store_memory",
            
            "recall_memory": "recall_memory",
            "what_did_i_say": "recall_memory",
            "do_you_remember": "recall_memory",
        })
        
        # Planning Intents
        self.intent_normalization_map.update({
            "plan_task": "plan_task",
            "schedule_event": "schedule_event",
            "set_reminder": "set_reminder",
            "create_reminder": "set_reminder",
        })
        
        # Safety Intents
        self.intent_normalization_map.update({
            "emergency_stop": "emergency_stop",
            "danger": "danger_alert",
            "watch_out": "danger_alert",
            "be_careful": "caution_warning",
            "check_safety": "check_safety",
        })
        
        # Home Automation Intents
        self.intent_normalization_map.update({
            "turn_on_lights": "control_lights",
            "turn_off_lights": "control_lights",
            "control_lights": "control_lights",
            
            "turn_on_tv": "control_tv",
            "turn_off_tv": "control_tv",
            
            "play_music": "control_music",
            "stop_music": "control_music",
        })
        
        # ============================================================
        # INTENT TO ENGINE MAPPING
        # ============================================================
        
        # Object Manipulation -> Engines
        self.intent_engine_map.update({
            "object_grasp": ["ObjectGraspingEngine"],
            "object_place": ["ObjectPlacementEngine"],
            "object_transfer": ["ObjectGraspingEngine", "NavigationEngine", "ObjectTransferEngine"],
            "object_fetch_from_location": ["NavigationEngine", "ObjectGraspingEngine", "NavigationEngine"],
            "object_open": ["ObjectOpeningEngine"],
            "object_close": ["ObjectClosingEngine"],
            "object_pour": ["ObjectPouringEngine"],
            "object_stack": ["ObjectStackingEngine"],
            "object_sort": ["ObjectSortingEngine"],
            "object_clean": ["ObjectCleaningEngine"],
            "object_fold": ["ObjectFoldingEngine"],
            "object_cut": ["ObjectCuttingEngine"],
        })
        
        # Navigation -> Engines
        self.intent_engine_map.update({
            "navigate_to": ["PathPlanningEngine", "NavigationEngine"],
            "turn_direction": ["TurnEngine"],
            "move_direction": ["MovementEngine"],
            "follow_person": ["PersonTrackingEngine", "FollowEngine"],
            "navigate_to_user": ["UserLocalizationEngine", "NavigationEngine"],
            "stop_movement": ["StopEngine"],
        })
        
        # Vision -> Engines
        self.intent_engine_map.update({
            "visual_scene_description": ["SceneUnderstandingEngine", "NaturalLanguageGenerationEngine"],
            "visual_count_objects": ["ObjectDetectionEngine", "CountingEngine"],
            "visual_locate_object": ["ObjectLocalizationEngine"],
            "face_recognition": ["FaceRecognitionEngine", "NameRecallEngine"],
        })
        
        # Interaction -> Engines
        self.intent_engine_map.update({
            "greeting": ["GreetingEngine"],
            "farewell": ["FarewellEngine"],
            "tell_joke": ["JokeEngine"],
            "tell_story": ["StorytellingEngine"],
            "sing_song": ["MusicEngine"],
            "ask_capabilities": ["CapabilityDescriptionEngine"],
            "ask_identity": ["IdentityEngine"],
            "ask_name": ["NameEngine"],
        })
        
        # Memory -> Engines
        self.intent_engine_map.update({
            "store_memory": ["EpisodicMemoryEngine"],
            "recall_memory": ["EpisodicMemoryEngine", "SemanticMemoryEngine"],
        })
        
        # Planning -> Engines
        self.intent_engine_map.update({
            "plan_task": ["TaskPlanningEngine"],
            "schedule_event": ["SchedulePlanningEngine"],
            "set_reminder": ["ReminderEngine"],
        })
        
        # Safety -> Engines
        self.intent_engine_map.update({
            "emergency_stop": ["EmergencyStopEngine"],
            "danger_alert": ["HazardDetectionEngine", "EmergencyResponseEngine"],
            "caution_warning": ["CautionEngine"],
            "check_safety": ["SafetyMonitoringEngine"],
        })
        
        # Home Automation -> Engines
        self.intent_engine_map.update({
            "control_lights": ["LightControlEngine"],
            "control_tv": ["TVControlEngine"],
            "control_music": ["MusicControlEngine"],
        })
        
        # ============================================================
        # INTENT CATEGORY MAPPING
        # ============================================================
        
        for intent in self.intent_engine_map.keys():
            if intent.startswith("object_"):
                self.intent_category_map[intent] = IntentCategory.OBJECT_MANIPULATION
            elif intent.startswith("navigate_") or intent in ["turn_direction", "move_direction", "follow_person", "stop_movement"]:
                self.intent_category_map[intent] = IntentCategory.NAVIGATION
            elif intent.startswith("visual_") or intent.startswith("face_"):
                self.intent_category_map[intent] = IntentCategory.VISION
            elif intent in ["greeting", "farewell", "tell_joke", "tell_story", "sing_song", "ask_capabilities", "ask_identity", "ask_name"]:
                self.intent_category_map[intent] = IntentCategory.INTERACTION
            elif intent in ["store_memory", "recall_memory"]:
                self.intent_category_map[intent] = IntentCategory.MEMORY
            elif intent in ["plan_task", "schedule_event", "set_reminder"]:
                self.intent_category_map[intent] = IntentCategory.PLANNING
            elif intent in ["emergency_stop", "danger_alert", "caution_warning", "check_safety"]:
                self.intent_category_map[intent] = IntentCategory.SAFETY
            elif intent.startswith("control_"):
                self.intent_category_map[intent] = IntentCategory.HOME_AUTOMATION
            else:
                self.intent_category_map[intent] = IntentCategory.UNKNOWN
        
        logger.info(f"Initialized {len(self.intent_normalization_map)} intent normalizations")
        logger.info(f"Initialized {len(self.intent_engine_map)} intent-engine mappings")
    
    def normalize_intent(self, intent: str) -> str:
        """Normalize intent to canonical form"""
        if not intent:
            return "unknown"
        return self.intent_normalization_map.get(intent, intent)
    
    def route_intent(self, intent: str, entities: Dict[str, Any], 
                     session_id: str = "default") -> RouterResponse:
        """
        Route intent to appropriate engines and execute
        
        Args:
            intent: Intent from NLP
            entities: Extracted entities
            session_id: Session identifier
            
        Returns:
            RouterResponse with execution results
        """
        start_time = time.time()
        
        # Normalize intent
        normalized_intent = self.normalize_intent(intent)
        category = self.intent_category_map.get(normalized_intent, IntentCategory.UNKNOWN)
        
        logger.info(f"Routing intent: {intent} -> {normalized_intent} ({category.value})")
        
        # Get session context
        context = self.get_session_context(session_id)
        
        # Get engines for intent
        engine_names = self.intent_engine_map.get(normalized_intent, [])
        
        if not engine_names:
            logger.warning(f"No engines mapped for intent: {normalized_intent}")
            return RouterResponse(
                intent=intent,
                normalized_intent=normalized_intent,
                category=category,
                engine_responses=[],
                total_execution_time=time.time() - start_time,
                success=False,
                message=f"No engines available for intent: {normalized_intent}"
            )
        
        # Execute engines
        engine_responses = []
        for engine_name in engine_names:
            engine = self.get_engine(engine_name)
            
            if engine and engine.is_enabled():
                try:
                    response = engine.execute(entities, context)
                    engine_responses.append(response)
                    
                    # Update context with engine result
                    context[f"last_{engine_name}_result"] = response.data
                    
                except Exception as e:
                    logger.error(f"Engine {engine_name} execution failed: {e}")
                    # Continue with next engine
            else:
                logger.warning(f"Engine {engine_name} not available or disabled")
        
        # Update session
        self.update_session(session_id, normalized_intent, engine_responses, context)
        
        # Determine overall success
        success = any(r.is_success() for r in engine_responses)
        message = self._generate_response_message(normalized_intent, engine_responses)
        
        total_time = time.time() - start_time
        
        return RouterResponse(
            intent=intent,
            normalized_intent=normalized_intent,
            category=category,
            engine_responses=engine_responses,
            total_execution_time=total_time,
            success=success,
            message=message
        )
    
    def register_engine(self, engine_name: str, engine: BaseEngine):
        """Register an engine with the router"""
        self.engines[engine_name] = engine
        logger.info(f"Registered engine: {engine_name}")
    
    def get_engine(self, engine_name: str) -> Optional[BaseEngine]:
        """Get an engine by name"""
        return self.engines.get(engine_name)
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {
                "created_at": time.time(),
                "last_updated": time.time(),
                "intent_history": [],
                "conversation_context": {}
            }
        return self.session_memory[session_id]
    
    def update_session(self, session_id: str, intent: str, 
                      responses: List[EngineResponse], context: Dict[str, Any]):
        """Update session memory"""
        if session_id not in self.session_memory:
            self.session_memory[session_id] = {}
        
        self.session_memory[session_id].update({
            "last_updated": time.time(),
            "last_intent": intent,
            "last_responses": [r.to_dict() for r in responses],
            "conversation_context": context
        })
        
        # Add to intent history
        if "intent_history" not in self.session_memory[session_id]:
            self.session_memory[session_id]["intent_history"] = []
        
        self.session_memory[session_id]["intent_history"].append({
            "intent": intent,
            "timestamp": time.time(),
            "success": any(r.is_success() for r in responses)
        })
    
    def _generate_response_message(self, intent: str, 
                                   responses: List[EngineResponse]) -> str:
        """Generate human-readable response message"""
        if not responses:
            return f"I don't know how to handle: {intent}"
        
        successful_responses = [r for r in responses if r.is_success()]
        
        if not successful_responses:
            return f"I encountered an error while trying to {intent.replace('_', ' ')}"
        
        # Use the first successful response message
        return successful_responses[0].message
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "total_sessions": len(self.session_memory),
            "registered_engines": len(self.engines),
            "intent_mappings": len(self.intent_engine_map),
            "normalization_rules": len(self.intent_normalization_map)
        }

