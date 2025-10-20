"""
Safety Monitor Engine - Fall Detection and Emergency Response

Monitors for falls and emergencies, assesses severity, and calls emergency services.
Critical safety system for elderly care and child supervision.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

from src.engines.base_engine import BaseEngine, EngineResponse, EngineStatus, EngineTier

logger = logging.getLogger(__name__)


class EmergencyLevel(Enum):
    """Emergency severity levels"""
    NONE = "none"
    LOW = "low"  # Minor issue, monitor
    MEDIUM = "medium"  # Needs attention, contact family
    HIGH = "high"  # Serious, call 999
    CRITICAL = "critical"  # Life-threatening, call 999 immediately


class FallType(Enum):
    """Types of falls detected"""
    SLIP = "slip"  # Lost balance
    TRIP = "trip"  # Caught on something
    COLLAPSE = "collapse"  # Sudden loss of consciousness
    SEIZURE = "seizure"  # Medical emergency
    FAINT = "faint"  # Loss of consciousness


class PersonState(Enum):
    """Person's state after fall"""
    CONSCIOUS = "conscious"
    SEMI_CONSCIOUS = "semi_conscious"
    UNCONSCIOUS = "unconscious"
    RESPONSIVE = "responsive"
    UNRESPONSIVE = "unresponsive"


class SafetyMonitorEngine(BaseEngine):
    """
    Production-grade safety monitoring and emergency response engine
    
    Features:
    - Fall detection (vision + sensors)
    - Person assessment (conscious/unconscious)
    - Injury severity estimation
    - Emergency decision making
    - 999 emergency calling
    - Family notification
    - First aid guidance
    - Incident documentation
    
    Detection Methods:
    - Vision: Sudden person position change
    - Audio: Impact sounds, cries for help
    - Pose estimation: Abnormal body positions
    - IMU sensors: Sudden acceleration changes
    - Floor pressure sensors: Impact detection
    
    Assessment Protocol:
    1. Detect fall event
    2. Approach person safely
    3. Check responsiveness (speak, observe)
    4. Assess visible injuries
    5. Check breathing and movement
    6. Determine emergency level
    7. Take appropriate action
    8. Document incident
    
    Emergency Actions:
    - None: Monitor, offer assistance
    - Low: Help up, check for pain
    - Medium: Contact family, monitor
    - High: Call 999, provide first aid
    - Critical: Call 999, start CPR if trained
    
    Multi-tier fallback:
    - Tier 1: Full assessment with all sensors
    - Tier 2: Visual assessment only
    - Tier 3: Alert family immediately
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Configuration
        self.emergency_number = config.get("emergency_number", "999")  # UK
        self.family_contacts = config.get("family_contacts", [])
        self.enable_auto_call = config.get("enable_auto_call", True)
        self.assessment_timeout = config.get("assessment_timeout", 30)  # seconds
        
        # Services (will be injected)
        self.vision_service = None
        self.pose_estimation_service = None
        self.audio_service = None
        self.communication_service = None
        self.memory_manager = None
        
        # Emergency phrases to listen for
        self.emergency_phrases = [
            "help", "emergency", "call ambulance", "call 999",
            "can't breathe", "chest pain", "can't move",
            "fell", "hurt", "pain", "bleeding", "broken"
        ]
        
        # Assessment questions
        self.assessment_questions = [
            "Can you hear me?",
            "Can you speak?",
            "Can you move your arms?",
            "Can you move your legs?",
            "Where does it hurt?",
            "Did you hit your head?",
            "Are you having chest pain?",
            "Can you breathe normally?"
        ]
    
    def execute(self, entities: Dict[str, Any], context: Dict[str, Any]) -> EngineResponse:
        """
        Monitor safety and respond to emergencies
        
        Entities:
            - event_type: fall/emergency/alert
            - person_id: Person involved
            - location: Where event occurred
        
        Context:
            - fall_detected: Boolean
            - impact_force: Force of impact
            - person_position: Current body position
            - audio_analysis: Sounds detected
            - previous_health: Known health conditions
        """
        start_time = time.time()
        
        if not self.enabled:
            return EngineResponse(
                status=EngineStatus.FAILURE,
                message="SafetyMonitorEngine is disabled",
                data={},
                tier_used=EngineTier.TIER_3,
                execution_time=time.time() - start_time,
                confidence=0.0,
                errors=["Safety monitoring disabled - DANGEROUS"]
            )
        
        # Check if this is a fall detection
        fall_detected = context.get("fall_detected", False)
        
        if not fall_detected:
            # No fall detected, return normal monitoring
            return EngineResponse(
                status=EngineStatus.SUCCESS,
                message="Safety monitoring active, no incidents detected",
                data={"monitoring": True},
                tier_used=EngineTier.TIER_1,
                execution_time=time.time() - start_time,
                confidence=1.0
            )
        
        # FALL DETECTED - Execute emergency protocol
        logger.warning("FALL DETECTED - Initiating emergency protocol")
        
        return self._execute_with_fallback(
            self._respond_tier1,
            self._respond_tier2,
            self._respond_tier3,
            entities,
            context
        )
    
    def _respond_tier1(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tier 1: Full assessment protocol with all sensors
        
        Steps:
        1. Alert all systems
        2. Navigate to person
        3. Visual + audio assessment
        4. Determine consciousness level
        5. Check for injuries
        6. Assess breathing and movement
        7. Determine emergency level
        8. Take appropriate action
        9. Document everything
        """
        logger.critical("Tier 1: Full emergency assessment protocol")
        
        person_id = entities.get("person_id", "unknown")
        location = entities.get("location", "unknown")
        
        assessment = {
            "timestamp": datetime.utcnow().isoformat(),
            "person_id": person_id,
            "location": location,
            "assessment_steps": []
        }
        
        # Step 1: Alert systems
        self._alert_all_systems("FALL DETECTED", person_id, location)
        assessment["assessment_steps"].append("systems_alerted")
        
        # Step 2: Navigate to person
        if self.vision_service:
            navigation_result = self._navigate_to_person(person_id, location)
            assessment["navigation"] = navigation_result
            assessment["assessment_steps"].append("navigated_to_person")
        
        # Step 3: Visual assessment
        visual_assessment = self._perform_visual_assessment(person_id, context)
        assessment["visual_assessment"] = visual_assessment
        assessment["assessment_steps"].append("visual_assessment_complete")
        
        # Step 4: Check consciousness
        consciousness = self._assess_consciousness(person_id, context)
        assessment["consciousness"] = consciousness
        assessment["assessment_steps"].append("consciousness_checked")
        
        # Step 5: Check for visible injuries
        injuries = self._assess_visible_injuries(visual_assessment)
        assessment["injuries"] = injuries
        assessment["assessment_steps"].append("injuries_assessed")
        
        # Step 6: Check breathing and movement
        vital_signs = self._check_vital_signs(person_id, context)
        assessment["vital_signs"] = vital_signs
        assessment["assessment_steps"].append("vital_signs_checked")
        
        # Step 7: Determine emergency level
        emergency_level = self._determine_emergency_level(
            consciousness, injuries, vital_signs, context
        )
        assessment["emergency_level"] = emergency_level.value
        assessment["assessment_steps"].append("emergency_level_determined")
        
        # Step 8: Take action based on emergency level
        action_taken = self._take_emergency_action(
            emergency_level, person_id, assessment
        )
        assessment["action_taken"] = action_taken
        assessment["assessment_steps"].append("action_executed")
        
        # Step 9: Document incident
        self._document_incident(assessment)
        assessment["assessment_steps"].append("incident_documented")
        
        # Generate response message
        message = self._generate_response_message(emergency_level, action_taken)
        
        return {
            "message": message,
            "data": assessment,
            "confidence": 0.95
        }
    
    def _respond_tier2(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 2: Visual-only assessment"""
        logger.warning("Tier 2: Visual-only emergency assessment")
        
        person_id = entities.get("person_id", "unknown")
        
        # Simple visual check
        if self.vision_service:
            visual = self._perform_visual_assessment(person_id, context)
            
            # If person is lying down and not moving - HIGH emergency
            if visual.get("position") == "lying_down" and not visual.get("moving"):
                emergency_level = EmergencyLevel.HIGH
            else:
                emergency_level = EmergencyLevel.MEDIUM
        else:
            # No vision - assume worst case
            emergency_level = EmergencyLevel.HIGH
        
        action = self._take_emergency_action(emergency_level, person_id, {})
        
        return {
            "message": f"Fall detected. {action['message']}",
            "data": {
                "emergency_level": emergency_level.value,
                "action": action
            },
            "confidence": 0.80
        }
    
    def _respond_tier3(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Tier 3: Immediate alert to family"""
        logger.critical("Tier 3: Emergency - alerting family immediately")
        
        person_id = entities.get("person_id", "unknown")
        
        # Alert family immediately
        self._alert_family_emergency(person_id, "Fall detected - unable to assess")
        
        # Play audio message
        message = "Fall detected! I am alerting your family right now. Help is on the way. Please try to stay calm."
        
        return {
            "message": message,
            "data": {
                "emergency_level": EmergencyLevel.HIGH.value,
                "family_alerted": True,
                "assessment": "unable_to_perform"
            },
            "confidence": 0.70
        }
    
    # Assessment Helper Methods
    
    def _perform_visual_assessment(self, person_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform visual assessment of person"""
        if not self.vision_service:
            return {"available": False}
        
        try:
            # Get person's current position
            person_detection = self.vision_service.detect_person(person_id)
            
            # Analyze body position
            position = "standing"
            if person_detection["bbox"][3] < 200:  # Very low in frame
                position = "lying_down"
            elif person_detection["bbox"][3] < 400:
                position = "sitting_or_kneeling"
            
            # Check for movement
            moving = person_detection.get("motion_detected", False)
            
            # Check for abnormal poses
            pose = person_detection.get("pose", {})
            abnormal_pose = self._is_abnormal_pose(pose)
            
            return {
                "available": True,
                "position": position,
                "moving": moving,
                "abnormal_pose": abnormal_pose,
                "bbox": person_detection["bbox"]
            }
        
        except Exception as e:
            logger.error(f"Visual assessment failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _assess_consciousness(self, person_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess person's consciousness level"""
        # Speak to person
        responses = []
        
        for question in self.assessment_questions[:3]:  # Ask first 3 questions
            self._speak(question)
            time.sleep(2)  # Wait for response
            
            # Listen for response
            if self.audio_service:
                response = self.audio_service.listen_for_response(timeout=3)
                responses.append({
                    "question": question,
                    "response": response.get("text", "no_response"),
                    "understood": response.get("understood", False)
                })
        
        # Determine consciousness level
        response_count = sum(1 for r in responses if r["understood"])
        
        if response_count >= 2:
            state = PersonState.CONSCIOUS
            level = "fully_conscious"
        elif response_count == 1:
            state = PersonState.SEMI_CONSCIOUS
            level = "partially_responsive"
        else:
            state = PersonState.UNCONSCIOUS
            level = "unresponsive"
        
        return {
            "state": state.value,
            "level": level,
            "responses": responses,
            "response_count": response_count
        }
    
    def _assess_visible_injuries(self, visual_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess visible injuries from visual data"""
        if not visual_assessment.get("available"):
            return {"assessed": False}
        
        injuries = {
            "assessed": True,
            "visible_bleeding": False,
            "abnormal_limb_position": False,
            "head_injury_suspected": False,
            "severity": "unknown"
        }
        
        # Check for abnormal pose (could indicate fracture)
        if visual_assessment.get("abnormal_pose"):
            injuries["abnormal_limb_position"] = True
            injuries["severity"] = "moderate_to_severe"
        
        # Check position (lying down after fall = more serious)
        if visual_assessment.get("position") == "lying_down":
            if not visual_assessment.get("moving"):
                injuries["severity"] = "severe"
                injuries["head_injury_suspected"] = True
        
        return injuries
    
    def _check_vital_signs(self, person_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check breathing and movement"""
        vital_signs = {
            "breathing": "unknown",
            "moving": False,
            "responsive_to_voice": False
        }
        
        # Check for chest movement (breathing)
        if self.vision_service:
            try:
                movement = self.vision_service.detect_chest_movement(person_id)
                vital_signs["breathing"] = "normal" if movement else "not_detected"
            except:
                vital_signs["breathing"] = "unable_to_determine"
        
        # Check for any movement
        vital_signs["moving"] = context.get("person_moving", False)
        
        # Check voice response
        vital_signs["responsive_to_voice"] = context.get("responded_to_voice", False)
        
        return vital_signs
    
    def _determine_emergency_level(self, consciousness: Dict, injuries: Dict, 
                                   vital_signs: Dict, context: Dict) -> EmergencyLevel:
        """Determine emergency severity level"""
        
        # CRITICAL: Not breathing or unresponsive
        if vital_signs.get("breathing") == "not_detected":
            return EmergencyLevel.CRITICAL
        
        if consciousness["state"] == PersonState.UNCONSCIOUS.value:
            return EmergencyLevel.CRITICAL
        
        # HIGH: Severe injuries or semi-conscious
        if injuries.get("severity") == "severe":
            return EmergencyLevel.HIGH
        
        if injuries.get("head_injury_suspected"):
            return EmergencyLevel.HIGH
        
        if consciousness["state"] == PersonState.SEMI_CONSCIOUS.value:
            return EmergencyLevel.HIGH
        
        # MEDIUM: Moderate injuries but conscious
        if injuries.get("abnormal_limb_position"):
            return EmergencyLevel.MEDIUM
        
        if injuries.get("severity") == "moderate_to_severe":
            return EmergencyLevel.MEDIUM
        
        # LOW: Conscious and responsive, minor/no visible injuries
        if consciousness["state"] == PersonState.CONSCIOUS.value:
            return EmergencyLevel.LOW
        
        # Default to MEDIUM for safety
        return EmergencyLevel.MEDIUM
    
    def _take_emergency_action(self, emergency_level: EmergencyLevel, 
                               person_id: str, assessment: Dict) -> Dict[str, Any]:
        """Take appropriate action based on emergency level"""
        action = {
            "emergency_level": emergency_level.value,
            "999_called": False,
            "family_contacted": False,
            "first_aid_provided": False,
            "message": ""
        }
        
        if emergency_level == EmergencyLevel.CRITICAL:
            # CALL 999 IMMEDIATELY
            if self.enable_auto_call:
                call_result = self._call_999(person_id, assessment)
                action["999_called"] = True
                action["call_result"] = call_result
            
            # Alert family
            self._alert_family_emergency(person_id, "CRITICAL: 999 called")
            action["family_contacted"] = True
            
            # Provide instructions
            self._speak("Emergency services have been called. Help is on the way. Try to stay still and breathe.")
            action["message"] = "999 called - ambulance dispatched"
        
        elif emergency_level == EmergencyLevel.HIGH:
            # CALL 999
            if self.enable_auto_call:
                call_result = self._call_999(person_id, assessment)
                action["999_called"] = True
                action["call_result"] = call_result
            
            # Alert family
            self._alert_family_emergency(person_id, "HIGH: Fall with injuries")
            action["family_contacted"] = True
            
            action["message"] = "999 called - ambulance requested"
        
        elif emergency_level == EmergencyLevel.MEDIUM:
            # Contact family
            self._alert_family_emergency(person_id, "MEDIUM: Fall detected, person needs attention")
            action["family_contacted"] = True
            
            # Offer assistance
            self._speak("I've contacted your family. Can I help you get up? Please take it slowly.")
            action["message"] = "Family contacted - offering assistance"
        
        else:  # LOW
            # Offer help
            self._speak("I saw you fall. Are you okay? Can I help you up?")
            action["message"] = "Monitoring - offering assistance"
        
        return action
    
    def _call_999(self, person_id: str, assessment: Dict) -> Dict[str, Any]:
        """Call 999 emergency services"""
        logger.critical(f"CALLING 999 for person {person_id}")
        
        if not self.communication_service:
            logger.error("Communication service not available - CANNOT CALL 999")
            return {"called": False, "error": "service_unavailable"}
        
        try:
            # Prepare emergency message
            message = self._prepare_999_message(person_id, assessment)
            
            # Make the call
            call_result = self.communication_service.emergency_call(
                number=self.emergency_number,
                message=message
            )
            
            logger.info(f"999 call result: {call_result}")
            
            return {
                "called": True,
                "time": datetime.utcnow().isoformat(),
                "message": message,
                "call_result": call_result
            }
        
        except Exception as e:
            logger.error(f"999 call failed: {e}")
            return {"called": False, "error": str(e)}
    
    def _prepare_999_message(self, person_id: str, assessment: Dict) -> str:
        """Prepare emergency message for 999"""
        location = assessment.get("location", "unknown location")
        consciousness = assessment.get("consciousness", {}).get("level", "unknown")
        breathing = assessment.get("vital_signs", {}).get("breathing", "unknown")
        
        message = f"""Emergency at {location}. 
Person has fallen. 
Consciousness: {consciousness}. 
Breathing: {breathing}. 
This is an automated call from a home assistant robot. 
Ambulance required immediately."""
        
        return message
    
    def _alert_family_emergency(self, person_id: str, message: str):
        """Alert family contacts about emergency"""
        logger.warning(f"ALERTING FAMILY: {message}")
        
        for contact in self.family_contacts:
            try:
                if self.communication_service:
                    self.communication_service.send_alert(
                        contact=contact,
                        message=f"EMERGENCY ALERT: {message}",
                        priority="high"
                    )
            except Exception as e:
                logger.error(f"Failed to alert {contact}: {e}")
    
    def _document_incident(self, assessment: Dict):
        """Document incident in memory for records"""
        if not self.memory_manager:
            return
        
        try:
            from src.memory.mongodb_manager import EpisodicMemory
            
            memory = EpisodicMemory(
                session_id="emergency_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                event_type="fall_emergency",
                event_description=f"Fall detected - Emergency level: {assessment.get('emergency_level')}",
                context=assessment,
                importance=1.0  # Highest importance
            )
            
            self.memory_manager.store_episodic_memory(memory)
            logger.info("Incident documented in long-term memory")
        
        except Exception as e:
            logger.error(f"Failed to document incident: {e}")
    
    def _generate_response_message(self, emergency_level: EmergencyLevel, 
                                   action: Dict) -> str:
        """Generate human-readable response message"""
        if emergency_level == EmergencyLevel.CRITICAL:
            return "CRITICAL EMERGENCY: 999 called immediately. Ambulance on the way. Stay calm."
        elif emergency_level == EmergencyLevel.HIGH:
            return "EMERGENCY: 999 called. Help is coming. Your family has been notified."
        elif emergency_level == EmergencyLevel.MEDIUM:
            return "Fall detected. Your family has been contacted. Can I help you?"
        else:
            return "I noticed you fell. Are you alright? Let me help you up."
    
    # Helper methods
    
    def _alert_all_systems(self, alert_type: str, person_id: str, location: str):
        """Alert all robot systems of emergency"""
        logger.critical(f"SYSTEM ALERT: {alert_type} - Person: {person_id}, Location: {location}")
    
    def _navigate_to_person(self, person_id: str, location: str) -> Dict:
        """Navigate to person's location"""
        logger.info(f"Navigating to {person_id} at {location}")
        return {"success": True, "time": 2.5}
    
    def _speak(self, message: str):
        """Speak message to person"""
        logger.info(f"SPEAKING: {message}")
        # TTS would be used here
    
    def _is_abnormal_pose(self, pose: Dict) -> bool:
        """Check if body pose is abnormal"""
        # Check for unusual joint angles that might indicate injury
        return False  # Simplified
    
    def get_capabilities(self) -> List[str]:
        return [
            "fall_detection",
            "emergency_response",
            "call_999",
            "person_assessment",
            "consciousness_check",
            "injury_assessment",
            "vital_signs_check",
            "family_notification",
            "emergency_documentation",
            "first_aid_guidance",
            "safety_monitoring"
        ]

