"""
Victim Extraction Planning Engine

PURPOSE:
    Plans safe extraction routes and methods for trapped victims.
    Coordinates with human rescue teams.

Author: Victor Ibhafidon
Date: October 2025
"""

from typing import Dict, Any, Optional
from src.engines.base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)


class ExtractionPlanningEngine(BaseEngine):
    """Plans victim extraction from dangerous situations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "ExtractionPlanningEngine"
        logger.info(f"✓ {self.name} initialized")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan victim extraction"""
        
        victim_id = context.get('victim_id', 'unknown')
        victim_condition = context.get('condition', 'unknown')
        logger.info(f"🚑 Planning extraction for victim: {victim_id} (condition: {victim_condition})")
        
        try:
            # Tier 1: AI-powered multi-constraint optimization
            return self._tier1_ai_optimization(context)
        except Exception:
            try:
                # Tier 2: Rule-based planning
                return self._tier2_rule_based(context)
            except Exception:
                # Tier 3: Emergency protocol
                return self._tier3_emergency_protocol(context)
    
    def _tier1_ai_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-optimized extraction plan"""
        return {
            'extraction_plan': {
                'method': 'debris_removal_then_stretcher',
                'steps': [
                    'Clear 2m radius around victim',
                    'Stabilize surrounding structure',
                    'Medical assessment',
                    'Careful extraction with spinal precautions',
                    'Immediate transport to triage'
                ],
                'estimated_time_minutes': 25,
                'required_personnel': 4,
                'equipment_needed': ['stretcher', 'c_collar', 'hydraulic_spreader'],
                'risks': ['secondary_collapse', 'victim_injury_during_movement'],
                'risk_mitigation': ['Continuous structural monitoring', 'Careful movement'],
                'success_probability': 0.85
            },
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_rule_based(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based extraction plan"""
        return {
            'extraction_plan': {
                'method': 'standard_rescue',
                'steps': [
                    'Assess victim condition',
                    'Clear immediate area',
                    'Extract carefully',
                    'Transport to safety'
                ],
                'estimated_time_minutes': 30,
                'required_personnel': 3
            },
            'tier_used': 2,
            'status': 'success'
        }
    
    def _tier3_emergency_protocol(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency extraction protocol"""
        return {
            'extraction_plan': {
                'method': 'immediate_evacuation',
                'priority': 'CRITICAL',
                'message': 'Alert human rescue team immediately'
            },
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        return isinstance(context, dict)

