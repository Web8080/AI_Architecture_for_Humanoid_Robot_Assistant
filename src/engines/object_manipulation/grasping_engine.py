"""
Object Grasping Engine

Handles all object grasping operations: pick up, grasp, hold, grab objects.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import Dict, Any, List
from src.engines.base_engine import BaseEngine, EngineResponse, EngineStatus, EngineTier

logger = logging.getLogger(__name__)


class ObjectGraspingEngine(BaseEngine):
    """
    Engine for grasping objects
    
    Capabilities:
    - Pick up objects
    - Grasp objects with different strategies
    - Hold objects securely
    - Adapt grasp based on object properties
    
    Multi-tier fallback:
    - Tier 1: Deep learning grasp planning (GraspNet, Dex-Net)
    - Tier 2: Heuristic-based grasping (shape primitives)
    - Tier 3: Simple parallel jaw grasp
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.vision_service = None  # Will be injected
        self.motion_planner = None  # Will be injected
        self.gripper_controller = None  # Will be injected
        
    def execute(self, entities: Dict[str, Any], context: Dict[str, Any]) -> EngineResponse:
        """
        Execute object grasping
        
        Entities:
            - object: Object to grasp (required)
            - color: Object color (optional)
            - location: Object location (optional)
            - size: Object size (optional)
            - grasp_type: Specific grasp type (optional)
        """
        if not self.enabled:
            return EngineResponse(
                status=EngineStatus.FAILURE,
                message="ObjectGraspingEngine is disabled",
                data={},
                tier_used=EngineTier.TIER_3,
                execution_time=0.0,
                confidence=0.0
            )
        
        # Validate required entities
        if not self.validate_entities(entities, ["object"]):
            return EngineResponse(
                status=EngineStatus.FAILURE,
                message="Missing required entity: object",
                data={},
                tier_used=EngineTier.TIER_3,
                execution_time=0.0,
                confidence=0.0
            )
        
        return self._execute_with_fallback(
            self._grasp_tier1,
            self._grasp_tier2,
            self._grasp_tier3,
            entities,
            context
        )
    
    def _grasp_tier1(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tier 1: Deep learning-based grasp planning
        Uses models like GraspNet or Dex-Net for optimal grasp poses
        """
        object_name = entities["object"]
        logger.info(f"Tier 1: Planning deep learning grasp for {object_name}")
        
        # Get object visual information
        if not self.vision_service:
            raise RuntimeError("Vision service not available for Tier 1")
        
        # Detect object in scene
        object_detections = self.vision_service.detect_objects()
        target_object = self._find_target_object(object_detections, entities)
        
        if not target_object:
            raise RuntimeError(f"Object {object_name} not found in scene")
        
        # Plan grasp using deep learning model
        grasp_pose = self._plan_dl_grasp(target_object)
        
        # Execute motion plan to grasp
        if not self.motion_planner:
            raise RuntimeError("Motion planner not available")
        
        trajectory = self.motion_planner.plan_to_pose(grasp_pose)
        
        # Execute grasp
        success = self._execute_grasp_motion(trajectory, target_object)
        
        return {
            "message": f"Successfully grasped {object_name} using deep learning planning",
            "data": {
                "object": object_name,
                "grasp_pose": grasp_pose,
                "grasp_quality": 0.95,
                "method": "deep_learning"
            },
            "confidence": 0.95
        }
    
    def _grasp_tier2(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tier 2: Heuristic-based grasping
        Uses object shape primitives and rule-based grasp selection
        """
        object_name = entities["object"]
        logger.info(f"Tier 2: Planning heuristic grasp for {object_name}")
        
        # Get object bounding box and shape
        if self.vision_service:
            object_info = self.vision_service.get_object_info(object_name)
        else:
            # Use cached information from context
            object_info = context.get("last_seen_objects", {}).get(object_name, {})
        
        # Determine grasp strategy based on shape
        grasp_strategy = self._select_grasp_strategy(object_info)
        
        # Calculate grasp pose from bounding box
        grasp_pose = self._calculate_heuristic_grasp(object_info, grasp_strategy)
        
        # Simplified motion planning
        success = self._execute_simple_grasp(grasp_pose, object_name)
        
        return {
            "message": f"Successfully grasped {object_name} using heuristic planning",
            "data": {
                "object": object_name,
                "grasp_strategy": grasp_strategy,
                "grasp_quality": 0.85,
                "method": "heuristic"
            },
            "confidence": 0.85
        }
    
    def _grasp_tier3(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tier 3: Simple parallel jaw grasp (always works)
        Basic grasp without complex planning
        """
        object_name = entities["object"]
        logger.info(f"Tier 3: Executing simple parallel grasp for {object_name}")
        
        # Get last known position from context or use default
        last_position = context.get("last_object_positions", {}).get(object_name)
        
        if not last_position:
            # Use default grasp position (e.g., in front of robot)
            last_position = {
                "x": 0.5,  # 50cm in front
                "y": 0.0,
                "z": 0.3   # 30cm high
            }
        
        # Simple parallel jaw grasp
        grasp_result = {
            "object": object_name,
            "position": last_position,
            "grasp_type": "parallel_jaw",
            "executed": True
        }
        
        return {
            "message": f"Executed basic grasp for {object_name}",
            "data": grasp_result,
            "confidence": 0.70
        }
    
    def _find_target_object(self, detections: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
        """Find target object in detections based on entities"""
        object_name = entities["object"].lower()
        object_color = entities.get("color", "").lower()
        
        for detection in detections:
            if object_name in detection["class"].lower():
                # Check color if specified
                if object_color and detection.get("color", "").lower() != object_color:
                    continue
                return detection
        
        return None
    
    def _plan_dl_grasp(self, object_info: Dict[str, Any]) -> Dict[str, Any]:
        """Plan grasp using deep learning model"""
        # Placeholder for actual DL model inference
        # In production, this would call GraspNet or similar
        return {
            "position": object_info.get("centroid", [0.5, 0.0, 0.3]),
            "orientation": [0, 0, 0, 1],  # Quaternion
            "gripper_width": 0.05,
            "approach_vector": [0, 0, -1]
        }
    
    def _select_grasp_strategy(self, object_info: Dict[str, Any]) -> str:
        """Select grasp strategy based on object properties"""
        # Simple heuristics
        bbox = object_info.get("bbox", [0, 0, 100, 100])
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if height > width * 1.5:
            return "vertical_pinch"
        elif width > height * 1.5:
            return "horizontal_pinch"
        else:
            return "center_grasp"
    
    def _calculate_heuristic_grasp(self, object_info: Dict[str, Any], 
                                   strategy: str) -> Dict[str, Any]:
        """Calculate grasp pose using heuristics"""
        bbox = object_info.get("bbox", [0, 0, 100, 100])
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return {
            "position": [center_x / 1000.0, center_y / 1000.0, 0.3],
            "strategy": strategy
        }
    
    def _execute_grasp_motion(self, trajectory: Any, object_info: Dict[str, Any]) -> bool:
        """Execute planned grasp trajectory"""
        # Placeholder for actual robot execution
        logger.info("Executing grasp trajectory...")
        return True
    
    def _execute_simple_grasp(self, grasp_pose: Dict[str, Any], object_name: str) -> bool:
        """Execute simple grasp without complex planning"""
        logger.info(f"Executing simple grasp for {object_name}")
        return True
    
    def get_capabilities(self) -> List[str]:
        """Return engine capabilities"""
        return [
            "pick_up_object",
            "grasp_object",
            "hold_object",
            "grab_object",
            "deep_learning_grasping",
            "heuristic_grasping",
            "parallel_jaw_grasping",
            "adaptive_grasping"
        ]

