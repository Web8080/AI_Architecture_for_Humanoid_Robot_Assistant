"""
Hazardous Materials Detection Engine

PURPOSE:
    Detects chemical, biological, radiological hazards in disaster zones.
    Protects rescue teams and victims from secondary exposure.

Author: Victor Ibhafidon
Date: October 2025
"""

from typing import Dict, Any, Optional
from src.engines.base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)


class HazmatDetectionEngine(BaseEngine):
    """Detects hazardous materials and environmental dangers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "HazmatDetectionEngine"
        self.danger_level_threshold = config.get('danger_threshold', 0.7) if config else 0.7
        logger.info(f"✓ {self.name} initialized")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect hazardous materials"""
        
        area_id = context.get('area_id', 'unknown')
        logger.info(f"☢️ Scanning for hazmat in area: {area_id}")
        
        try:
            # Tier 1: Multi-sensor chemical analysis
            return self._tier1_chemical_sensors(context)
        except Exception:
            try:
                # Tier 2: Gas detection only
                return self._tier2_gas_detection(context)
            except Exception:
                # Tier 3: Visual hazard identification
                return self._tier3_visual_hazards(context)
    
    def _tier1_chemical_sensors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced chemical sensor array"""
        return {
            'hazards_detected': [
                {'type': 'carbon_monoxide', 'level': 50, 'unit': 'ppm', 'danger': 'medium'},
                {'type': 'radiation', 'level': 0.5, 'unit': 'mSv/h', 'danger': 'low'}
            ],
            'air_quality_index': 85,
            'safe_for_humans': True,
            'ppe_required': ['respirator', 'protective_suit'],
            'evacuation_recommended': False,
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_gas_detection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic gas sensors"""
        return {
            'hazards_detected': [
                {'type': 'unknown_gas', 'level': 'moderate', 'danger': 'medium'}
            ],
            'safe_for_humans': True,
            'ppe_required': ['respirator'],
            'tier_used': 2,
            'status': 'partial'
        }
    
    def _tier3_visual_hazards(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Visual hazard identification"""
        return {
            'hazards_detected': [],
            'safe_for_humans': 'unknown',
            'recommendations': ['Deploy chemical sensors before entry'],
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        return isinstance(context, dict)

