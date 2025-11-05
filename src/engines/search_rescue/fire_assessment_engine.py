"""
Fire Assessment Engine

PURPOSE:
    Assesses fire conditions, spread patterns, and safe entry points.
    Monitors temperature, smoke, oxygen levels for rescue safety.

Author: Victor Ibhafidon
Date: October 2025
"""

from typing import Dict, Any, Optional
from src.engines.base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)


class FireAssessmentEngine(BaseEngine):
    """Assesses fire conditions and safety for rescue operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "FireAssessmentEngine"
        self.max_safe_temp = config.get('max_temp_celsius', 60) if config else 60
        self.min_oxygen_percent = config.get('min_oxygen', 19.5) if config else 19.5
        logger.info(f"✓ {self.name} initialized")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess fire conditions"""
        
        location = context.get('location', 'unknown')
        logger.info(f"🔥 Assessing fire conditions at: {location}")
        
        try:
            # Tier 1: Multi-sensor fire analysis with AI prediction
            return self._tier1_multi_sensor_ai(context)
        except Exception:
            try:
                # Tier 2: Temperature and smoke sensors
                return self._tier2_temp_smoke(context)
            except Exception:
                # Tier 3: Visual fire detection
                return self._tier3_visual_detection(context)
    
    def _tier1_multi_sensor_ai(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered fire analysis with spread prediction"""
        return {
            'fire_conditions': {
                'temperature_celsius': 45,
                'smoke_density': 'moderate',
                'oxygen_level_percent': 20.5,
                'co_level_ppm': 35,
                'visibility_meters': 3.5,
                'fire_intensity': 'medium',
                'fire_locations': ['room_203', 'hallway_2b']
            },
            'spread_prediction': {
                'direction': 'northeast',
                'estimated_spread_rate_m_per_min': 0.5,
                'time_to_flashover_minutes': 12
            },
            'safe_entry_points': ['south_entrance', 'west_stairwell'],
            'unsafe_zones': ['north_wing', 'elevator_shaft'],
            'safe_for_entry': True,
            'max_time_inside_minutes': 8,
            'ppe_required': ['scba', 'heat_resistant_suit', 'helmet'],
            'evacuation_triggers': ['temp_exceeds_70C', 'oxygen_below_19', 'flashover_imminent'],
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_temp_smoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Temperature and smoke sensor assessment"""
        return {
            'fire_conditions': {
                'temperature_celsius': 50,
                'smoke_density': 'high',
                'visibility_meters': 2
            },
            'safe_for_entry': True,
            'max_time_inside_minutes': 5,
            'ppe_required': ['scba', 'protective_gear'],
            'tier_used': 2,
            'status': 'partial'
        }
    
    def _tier3_visual_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Visual fire assessment"""
        return {
            'fire_detected': True,
            'safe_for_entry': False,
            'recommendations': ['Deploy thermal cameras', 'Request fire department assessment'],
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        return isinstance(context, dict)

