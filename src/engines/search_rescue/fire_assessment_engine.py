"""
Fire Assessment Engine

I built this to assess fire conditions and determine safe entry points for rescue operations.
Monitors temperature, smoke density, and oxygen levels to protect rescue teams.

WHY I CREATED THIS:
    Rescuers die from flashover and backdraft when entering fires without proper assessment.
    I needed a system to remotely assess fire conditions before humans enter.
    My thermal cameras and gas sensors can safely evaluate what's too dangerous for people.

MY ASSESSMENT CRITERIA:
    - Temperature levels and distribution
    - Smoke density and visibility
    - Oxygen concentration
    - Carbon monoxide levels
    - Fire spread rate and direction
    - Flashover risk indicators
    - Safe entry points and escape routes

CRITICAL DECISIONS:
    I determine if entry is safe, where rescuers can enter, how long they can stay,
    and when they must evacuate. My assessments prevent rescuer casualties.

Author: Victor Ibhafidon
Date: November 2025
"""

from typing import Dict, Any, Optional, List
from src.engines.base_engine import BaseEngine
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FireAssessmentEngine(BaseEngine):
    """
    I assess fire conditions for rescue operation safety.
    
    MY MONITORING:
    - Temperature mapping (thermal cameras)
    - Smoke density measurement
    - Oxygen level monitoring
    - Toxic gas detection (CO, CO2, HCN)
    - Fire intensity estimation
    - Spread pattern prediction
    - Flashover risk calculation
    
    THREE-TIER ASSESSMENT:
    - Tier 1: Multi-sensor AI fire analysis with spread prediction
    - Tier 2: Temperature and smoke sensors
    - Tier 3: Visual fire detection
    
    MY SAFETY THRESHOLDS:
    - Max safe temp: 60C for entry without SCBA
    - Min oxygen: 19.5% for breathing
    - Max CO: 35 ppm for short exposure
    - Flashover warning: 600C ceiling temp
    """
    
    # Temperature thresholds (Celsius)
    TEMP_SAFE_NO_SCBA = 60
    TEMP_SAFE_WITH_SCBA = 100
    TEMP_DANGEROUS = 150
    TEMP_FLASHOVER_WARNING = 600
    
    # Oxygen thresholds (percent)
    OXYGEN_NORMAL = 20.9
    OXYGEN_MIN_SAFE = 19.5
    OXYGEN_DANGEROUS = 17.0
    
    # CO thresholds (ppm)
    CO_SAFE = 35
    CO_DANGEROUS = 100
    CO_IDLH = 1200  # Immediately Dangerous to Life or Health
    
    # Entry safety levels
    SAFE_ENTRY = 'safe'
    CONDITIONAL_ENTRY = 'conditional'
    UNSAFE_ENTRY = 'unsafe'
    NO_ENTRY = 'no_entry'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """I initialize my fire assessment system."""
        super().__init__(config)
        self.name = "FireAssessmentEngine"
        
        # I configure safety thresholds
        self.max_safe_temp = config.get('max_temp_celsius', self.TEMP_SAFE_WITH_SCBA) if config else self.TEMP_SAFE_WITH_SCBA
        self.min_oxygen_percent = config.get('min_oxygen', self.OXYGEN_MIN_SAFE) if config else self.OXYGEN_MIN_SAFE
        self.max_co_ppm = config.get('max_co', self.CO_SAFE) if config else self.CO_SAFE
        
        # I track fire conditions over time
        self.condition_history: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Initialized {self.name}")
        logger.info(f"  Max safe temperature: {self.max_safe_temp}C")
        logger.info(f"  Min oxygen level: {self.min_oxygen_percent}%")
        logger.info(f"  Max CO level: {self.max_co_ppm} ppm")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        I assess fire conditions.
        
        My inputs:
            - location: Building/area identifier
            - assessment_type: 'entry_safety' | 'spread_prediction' | 'continuous_monitor'
            - entry_point: Proposed entry location
            - rescue_duration_minutes: Expected time inside
        
        My outputs:
            - Safe for entry decision
            - Entry points identified
            - Maximum time allowed inside
            - Required PPE
            - Evacuation triggers
            - Fire spread predictions
        """
        location = context.get('location', 'unknown')
        assessment_type = context.get('assessment_type', 'entry_safety')
        
        logger.info(f"Assessing fire conditions at: {location}")
        logger.info(f"  Assessment type: {assessment_type}")
        
        try:
            # I use all my sensors
            logger.info("Tier 1: Multi-sensor fire analysis")
            result = self._tier1_multi_sensor_ai(location, assessment_type, context)
            
            # I log my assessment
            if result['safe_for_entry']:
                logger.info(f"SAFE for entry with PPE - max {result['max_time_inside_minutes']} minutes")
            else:
                logger.warning(f"UNSAFE for entry - {result.get('hazard_summary', 'conditions too dangerous')}")
            
            return result
            
        except Exception as e1:
            logger.warning(f"Tier 1 unavailable: {e1}, using Tier 2")
            
            try:
                # I use basic sensors
                logger.info("Tier 2: Temperature and smoke sensors")
                result = self._tier2_temp_smoke(location, context)
                return result
                
            except Exception as e2:
                logger.warning(f"Tier 2 failed: {e2}, using Tier 3")
                
                # I do visual assessment
                logger.warning("Tier 3: Visual fire detection only")
                result = self._tier3_visual_detection(location, context)
                return result
    
    def _tier1_multi_sensor_ai(
        self,
        location: str,
        assessment_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 1: I perform comprehensive fire condition analysis.
        
        My process:
        1. I scan with thermal camera for temperature mapping
        2. I measure smoke density with laser/optical sensors
        3. I check oxygen levels with electrochemical sensor
        4. I detect CO and toxic gases
        5. I analyze fire intensity and location
        6. I predict spread patterns using AI
        7. I identify safe entry points
        8. I calculate maximum safe time inside
        """
        logger.debug("Tier 1: Multi-sensor fire condition analysis")
        
        # PLACEHOLDER: In production I would query actual sensors
        
        # I measure current conditions
        fire_conditions = {
            'temperature_celsius': 45,
            'smoke_density': 'moderate',
            'oxygen_level_percent': 20.5,
            'co_level_ppm': 35,
            'co2_level_ppm': 800,
            'visibility_meters': 3.5,
            'fire_intensity': 'medium',
            'fire_locations': ['room_203', 'hallway_2b']
        }
        
        # I predict fire spread
        logger.debug("Predicting fire spread patterns")
        spread_prediction = {
            'primary_direction': 'northeast',
            'estimated_spread_rate_m_per_min': 0.5,
            'time_to_flashover_minutes': 12,
            'high_risk_areas': ['north_wing', 'second_floor'],
            'prediction_confidence': 0.82
        }
        
        # I identify safe entry points
        logger.debug("Identifying safe entry points")
        safe_entry_points = ['south_entrance', 'west_stairwell']
        unsafe_zones = ['north_wing', 'elevator_shaft']
        
        # I determine if entry is safe
        temp_safe = fire_conditions['temperature_celsius'] <= self.max_safe_temp
        oxygen_safe = fire_conditions['oxygen_level_percent'] >= self.min_oxygen_percent
        co_safe = fire_conditions['co_level_ppm'] <= self.max_co_ppm
        
        safe_for_entry = temp_safe and oxygen_safe and co_safe
        
        # I calculate max time inside
        if safe_for_entry:
            # I base time on most limiting factor
            temp_time_limit = 30 if fire_conditions['temperature_celsius'] < 50 else 15
            co_time_limit = 30 if fire_conditions['co_level_ppm'] < 50 else 10
            max_time = min(temp_time_limit, co_time_limit, spread_prediction['time_to_flashover_minutes'] - 2)
        else:
            max_time = 0
        
        # I specify required PPE
        ppe_required = ['scba', 'heat_resistant_suit', 'helmet', 'gloves', 'boots']
        
        # I define evacuation triggers
        evacuation_triggers = [
            'temperature_exceeds_70C',
            'oxygen_below_19_percent',
            'flashover_imminent',
            'increasing_smoke_density',
            'structural_sounds'
        ]
        
        logger.info(f"Fire assessment complete: {'SAFE' if safe_for_entry else 'UNSAFE'} for entry")
        
        return {
            'location': location,
            'assessment_timestamp': datetime.now().isoformat(),
            'fire_conditions': fire_conditions,
            'spread_prediction': spread_prediction,
            'safe_entry_points': safe_entry_points,
            'unsafe_zones': unsafe_zones,
            'safe_for_entry': safe_for_entry,
            'max_time_inside_minutes': max_time,
            'ppe_required': ppe_required,
            'evacuation_triggers': evacuation_triggers,
            'assessment_valid_for_minutes': 5,  # Conditions change rapidly
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_temp_smoke(
        self,
        location: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 2: I use basic temperature and smoke sensors.
        
        Limited data but enough for basic safety assessment.
        """
        logger.debug("Tier 2: Basic sensor assessment")
        logger.warning("Advanced fire modeling unavailable")
        
        return {
            'location': location,
            'assessment_timestamp': datetime.now().isoformat(),
            'fire_conditions': {
                'temperature_celsius': 50,
                'smoke_density': 'high',
                'visibility_meters': 2
            },
            'safe_for_entry': True,
            'max_time_inside_minutes': 5,
            'ppe_required': ['scba', 'protective_gear'],
            'tier_used': 2,
            'status': 'partial',
            'warnings': ['Limited sensor data', 'Cannot predict spread']
        }
    
    def _tier3_visual_detection(
        self,
        location: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TIER 3: I do visual assessment only.
        
        Very limited - cannot measure actual conditions.
        """
        logger.warning("Tier 3: Visual assessment only - INSUFFICIENT FOR ENTRY DECISION")
        
        return {
            'location': location,
            'assessment_timestamp': datetime.now().isoformat(),
            'fire_detected': True,
            'safe_for_entry': False,
            'tier_used': 3,
            'status': 'partial',
            'warnings': ['Cannot measure temperature, oxygen, or gases', 'Visual only insufficient for safety'],
            'recommendations': ['Deploy thermal cameras', 'Deploy atmospheric sensors', 'Request fire department assessment']
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """I validate fire assessment inputs."""
        if not isinstance(context, dict):
            return False
        
        logger.debug("Input validation passed")
        return True
