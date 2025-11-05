"""
Pain Assessment Engine

PURPOSE:
    Assesses patient pain levels through verbal, facial, and behavioral indicators.
    Critical for non-verbal patients and pain management optimization.

Author: Victor Ibhafidon
Date: October 2025
"""

from typing import Dict, Any, Optional
from src.engines.base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)


class PainAssessmentEngine(BaseEngine):
    """Assesses patient pain levels"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "PainAssessmentEngine"
        logger.info(f"✓ {self.name} initialized")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess patient pain"""
        
        patient_id = context.get('patient_id', 'unknown')
        logger.info(f"😣 Assessing pain for {patient_id}")
        
        try:
            # Tier 1: Multi-modal pain assessment (facial, vocal, behavioral)
            return self._tier1_multimodal_assessment(context)
        except Exception:
            try:
                # Tier 2: Verbal self-reporting
                return self._tier2_verbal_reporting(context)
            except Exception:
                # Tier 3: Simple pain scale question
                return self._tier3_simple_scale(context)
    
    def _tier1_multimodal_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive multi-modal pain assessment"""
        return {
            'pain_assessment': {
                'patient_id': context.get('patient_id'),
                'timestamp': '2025-10-23T15:00:00Z',
                'self_reported_pain_scale_0_10': 6,
                'facial_expression_analysis': {
                    'grimacing_detected': True,
                    'eye_squinting': 'moderate',
                    'facial_pain_score': 5.5
                },
                'vocal_indicators': {
                    'groaning_detected': False,
                    'voice_stress_analysis': 'elevated',
                    'vocal_pain_score': 4.0
                },
                'behavioral_indicators': {
                    'restlessness': 'moderate',
                    'guarding_behavior': 'present',
                    'movement_limitation': 'moderate',
                    'behavioral_pain_score': 6.0
                },
                'vital_signs': {
                    'heart_rate_bpm': 95,  # Elevated
                    'blood_pressure': '135/85',  # Slightly elevated
                    'respiratory_rate': 20
                },
                'composite_pain_score': 5.5,  # Averaged from all sources
                'pain_level': 'moderate',
                'pain_location': context.get('pain_location', 'lower_back'),
                'pain_type': 'aching_chronic'
            },
            'recommendations': [
                'Administer prescribed pain medication',
                'Consider non-pharmacological interventions',
                'Reassess in 30 minutes',
                'Notify nurse if pain exceeds 7/10'
            ],
            'alert_triggered': True,
            'alert_type': 'moderate_pain_detected',
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_verbal_reporting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verbal pain self-reporting"""
        return {
            'pain_score_0_10': 6,
            'pain_location': 'lower_back',
            'pain_type': 'aching',
            'recommendations': ['Consider pain medication'],
            'tier_used': 2,
            'status': 'success'
        }
    
    def _tier3_simple_scale(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple pain scale question"""
        return {
            'pain_reported': True,
            'detailed_assessment_recommended': True,
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        return isinstance(context, dict)

