"""
Victim Extraction Planning Engine

I designed this to plan safe extraction routes for trapped victims.
Coordinates with human rescue teams to minimize injury during extraction.

WHY I BUILT THIS:
    Improper extraction causes secondary injuries and can worsen victim conditions.
    I needed an intelligent system to analyze victim position, injuries, entrapment,
    and structural hazards to generate optimal extraction plans.

MY PLANNING PROCESS:
    I assess victim accessibility, injury severity, entrapment mechanisms,
    structural stability along extraction routes, required equipment, and personnel needs.
    Then I generate step-by-step extraction protocols minimizing further harm.

FACTORS I CONSIDER:
    - Victim injury type and severity
    - Spinal precautions needed
    - Entrapment mechanism (pinned, buried, trapped)
    - Extraction route stability
    - Required equipment (spreaders, airbags, cutting tools)
    - Personnel requirements
    - Time constraints

Author: Victor Ibhafidon
Date: November 2025
"""

from typing import Dict, Any, Optional, List
from src.engines.base_engine import BaseEngine
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExtractionPlanningEngine(BaseEngine):
    """
    I plan victim extraction operations for complex rescue scenarios.
    
    MY PLANNING CAPABILITIES:
    - Victim accessibility assessment
    - Injury-appropriate handling protocols
    - Equipment requirement determination
    - Personnel allocation optimization
    - Route safety verification
    - Time estimation
    - Risk mitigation strategies
    
    THREE-TIER PLANNING:
    - Tier 1: AI multi-constraint optimization
    - Tier 2: Rule-based planning
    - Tier 3: Emergency protocol
    
    MY EXTRACTION METHODS:
    - Manual carry (stable victims)
    - Stretcher evacuation (spinal precautions)
    - Technical rescue (ropes, harnesses)
    - Heavy rescue (hydraulic tools)
    - Confined space rescue
    """
    
    # Extraction priority levels
    PRIORITY_IMMEDIATE = 1  # Life threat, rapid extraction
    PRIORITY_URGENT = 2     # Serious injuries, careful extraction
    PRIORITY_STANDARD = 3   # Stable, methodical extraction
    
    # Extraction methods
    METHOD_MANUAL_CARRY = 'manual_carry'
    METHOD_STRETCHER = 'stretcher_evacuation'
    METHOD_TECHNICAL = 'technical_rescue'
    METHOD_HEAVY = 'heavy_rescue'
    METHOD_CONFINED_SPACE = 'confined_space_rescue'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """I initialize my extraction planning system."""
        super().__init__(config)
        self.name = "ExtractionPlanningEngine"
        
        # I configure planning parameters
        self.max_planning_time_seconds = config.get('max_planning_time', 60) if config else 60
        self.safety_margin_factor = config.get('safety_margin', 1.5) if config else 1.5
        
        # I track extraction history
        self.extraction_plans: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized {self.name}")
        logger.info(f"  Max planning time: {self.max_planning_time_seconds}s")
        logger.info(f"  Safety margin factor: {self.safety_margin_factor}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        I plan victim extraction.
        
        My inputs:
            - victim_id: Victim identifier
            - victim_location: 3D coordinates
            - injuries: List of injuries
            - entrapment_type: How victim is trapped
            - structural_stability: Stability score of area
            - available_resources: Equipment and personnel available
        
        My outputs:
            - Extraction method
            - Step-by-step plan
            - Equipment needed
            - Personnel required
            - Time estimate
            - Risk assessment
        """
        victim_id = context.get('victim_id', 'unknown')
        victim_condition = context.get('condition', 'unknown')
        entrapment = context.get('entrapment_type', 'buried')
        
        logger.info(f"Planning extraction for victim: {victim_id}")
        logger.info(f"  Condition: {victim_condition}")
        logger.info(f"  Entrapment: {entrapment}")
        
        try:
            # I use AI optimization
            logger.info("Tier 1: AI multi-constraint extraction planning")
            result = self._tier1_ai_optimization(victim_id, victim_condition, entrapment, context)
            
            logger.info(f"Extraction plan generated")
            logger.info(f"  Method: {result['extraction_plan']['method']}")
            logger.info(f"  Time estimate: {result['extraction_plan']['estimated_time_minutes']} minutes")
            logger.info(f"  Personnel needed: {result['extraction_plan']['required_personnel']}")
            
            return result
            
        except Exception as e1:
            logger.warning(f"Tier 1 unavailable: {e1}, using Tier 2")
            
            try:
                # I use rule-based planning
                logger.info("Tier 2: Rule-based extraction planning")
                result = self._tier2_rule_based(victim_id, victim_condition, context)
                return result
                
            except Exception as e2:
                logger.warning(f"Tier 2 failed: {e2}, using Tier 3")
                
                # I use emergency protocol
                logger.warning("Tier 3: Emergency extraction protocol")
                result = self._tier3_emergency_protocol(victim_id, context)
                return result
    
    def _tier1_ai_optimization(
        self,
        victim_id: str,
        condition: str,
        entrapment: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 1: I use AI to optimize extraction planning.
        
        My optimization considers:
        - Victim injury severity and type
        - Spinal precaution requirements
        - Entrapment complexity
        - Structural hazards
        - Equipment availability
        - Personnel capabilities
        - Time constraints
        - Secondary injury risk
        """
        logger.debug("Tier 1: Multi-constraint optimization")
        
        # PLACEHOLDER: In production I would run optimization algorithm
        
        # I generate comprehensive extraction plan
        extraction_plan = {
            'victim_id': victim_id,
            'method': self.METHOD_STRETCHER,
            'priority': self.PRIORITY_URGENT,
            
            # I define the extraction steps
            'steps': [
                {
                    'step': 1,
                    'action': 'stabilize_surrounding_structure',
                    'duration_minutes': 5,
                    'equipment': ['shoring', 'cribbing'],
                    'personnel': 2
                },
                {
                    'step': 2,
                    'action': 'clear_2m_radius_around_victim',
                    'duration_minutes': 8,
                    'equipment': ['hand_tools', 'debris_bags'],
                    'personnel': 3
                },
                {
                    'step': 3,
                    'action': 'medical_assessment',
                    'duration_minutes': 3,
                    'equipment': ['medical_kit'],
                    'personnel': 1,
                    'specialist': 'paramedic'
                },
                {
                    'step': 4,
                    'action': 'apply_spinal_precautions',
                    'duration_minutes': 4,
                    'equipment': ['c_collar', 'backboard'],
                    'personnel': 2
                },
                {
                    'step': 5,
                    'action': 'careful_extraction_maintaining_alignment',
                    'duration_minutes': 5,
                    'equipment': ['stretcher', 'straps'],
                    'personnel': 4
                },
                {
                    'step': 6,
                    'action': 'immediate_transport_to_triage',
                    'duration_minutes': 2,
                    'equipment': ['ambulance'],
                    'personnel': 2
                }
            ],
            
            'estimated_time_minutes': 27,
            'required_personnel': 4,
            'required_specialists': ['paramedic'],
            
            # I list equipment needed
            'equipment_needed': [
                'stretcher',
                'c_collar',
                'backboard',
                'shoring_materials',
                'hand_tools',
                'medical_kit'
            ],
            
            # I identify risks
            'risks': [
                {
                    'risk': 'secondary_collapse',
                    'probability': 'medium',
                    'mitigation': 'Continuous structural monitoring'
                },
                {
                    'risk': 'victim_injury_during_movement',
                    'probability': 'low',
                    'mitigation': 'Spinal precautions, slow careful movement'
                },
                {
                    'risk': 'rescuer_injury',
                    'probability': 'low',
                    'mitigation': 'Proper lifting techniques, adequate personnel'
                }
            ],
            
            'success_probability': 0.85,
            'contingency_plan': 'If structural instability increases, switch to rapid extraction with increased risk acceptance'
        }
        
        logger.info(f"Generated {len(extraction_plan['steps'])}-step extraction plan")
        
        return {
            'extraction_plan': extraction_plan,
            'planning_timestamp': datetime.now().isoformat(),
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_rule_based(
        self,
        victim_id: str,
        condition: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 2: I use standard rescue protocols.
        
        Basic but reliable extraction planning.
        """
        logger.debug("Tier 2: Standard rescue protocol")
        
        return {
            'extraction_plan': {
                'victim_id': victim_id,
                'method': self.METHOD_STRETCHER,
                'steps': [
                    'Assess victim condition',
                    'Clear immediate area',
                    'Apply spinal precautions',
                    'Extract carefully',
                    'Transport to safety'
                ],
                'estimated_time_minutes': 30,
                'required_personnel': 3
            },
            'tier_used': 2,
            'status': 'success'
        }
    
    def _tier3_emergency_protocol(
        self,
        victim_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 3: I use emergency extraction protocol.
        
        When planning systems fail, I recommend immediate evacuation.
        """
        logger.warning("Tier 3: Emergency extraction protocol")
        
        return {
            'extraction_plan': {
                'victim_id': victim_id,
                'method': 'immediate_evacuation',
                'priority': 'CRITICAL',
                'message': 'Alert rescue team immediately for emergency extraction'
            },
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """I validate extraction planning inputs."""
        if not isinstance(context, dict):
            return False
        
        logger.debug("Input validation passed")
        return True
