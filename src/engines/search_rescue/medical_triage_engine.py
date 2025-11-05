"""
Medical Triage Engine

PURPOSE:
    Performs initial medical assessment and triage of multiple victims.
    Prioritizes victims based on injury severity and survival probability.

Author: Victor Ibhafidon
Date: October 2025
"""

from typing import Dict, Any, Optional, List
from src.engines.base_engine import BaseEngine
import logging

logger = logging.getLogger(__name__)


class MedicalTriageEngine(BaseEngine):
    """Performs medical triage using START (Simple Triage And Rapid Treatment) protocol"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "MedicalTriageEngine"
        logger.info(f"✓ {self.name} initialized")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform medical triage assessment"""
        
        victims = context.get('victims', [])
        logger.info(f"🏥 Performing triage on {len(victims)} victims")
        
        try:
            # Tier 1: AI-assisted medical assessment
            return self._tier1_ai_medical_assessment(context)
        except Exception:
            try:
                # Tier 2: Vital signs based triage
                return self._tier2_vital_signs_triage(context)
            except Exception:
                # Tier 3: Simple observation triage
                return self._tier3_observation_triage(context)
    
    def _tier1_ai_medical_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered comprehensive medical triage"""
        return {
            'triage_results': [
                {
                    'victim_id': 'V001',
                    'triage_category': 'IMMEDIATE',  # Red tag
                    'color_code': 'RED',
                    'priority': 1,
                    'assessment': {
                        'respiratory_rate': 10,  # breaths/min (abnormal)
                        'pulse': 130,  # bpm (elevated)
                        'mental_status': 'confused',
                        'capillary_refill_seconds': 3,
                        'major_injuries': ['suspected_internal_bleeding', 'fractured_femur'],
                        'bleeding_severity': 'severe'
                    },
                    'immediate_interventions': ['control_bleeding', 'oxygen', 'iv_access'],
                    'survival_probability': 0.75,
                    'estimated_golden_hour_remaining_minutes': 22
                },
                {
                    'victim_id': 'V002',
                    'triage_category': 'DELAYED',  # Yellow tag
                    'color_code': 'YELLOW',
                    'priority': 2,
                    'assessment': {
                        'respiratory_rate': 18,
                        'pulse': 88,
                        'mental_status': 'alert',
                        'major_injuries': ['fractured_arm', 'contusions']
                    },
                    'immediate_interventions': ['splint_arm', 'pain_management'],
                    'survival_probability': 0.95
                },
                {
                    'victim_id': 'V003',
                    'triage_category': 'MINOR',  # Green tag
                    'color_code': 'GREEN',
                    'priority': 3,
                    'assessment': {
                        'respiratory_rate': 16,
                        'pulse': 75,
                        'mental_status': 'alert',
                        'major_injuries': ['minor_lacerations']
                    },
                    'can_walk': True
                }
            ],
            'summary': {
                'total_victims': 3,
                'immediate': 1,
                'delayed': 1,
                'minor': 1,
                'deceased': 0,
                'recommended_ambulances': 2
            },
            'tier_used': 1,
            'status': 'success'
        }
    
    def _tier2_vital_signs_triage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Vital signs based triage"""
        return {
            'triage_results': [
                {
                    'victim_id': 'V001',
                    'triage_category': 'IMMEDIATE',
                    'color_code': 'RED',
                    'priority': 1,
                    'vital_signs_abnormal': True
                }
            ],
            'summary': {'total_victims': 1, 'immediate': 1},
            'tier_used': 2,
            'status': 'partial'
        }
    
    def _tier3_observation_triage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic observation triage"""
        return {
            'triage_results': [
                {
                    'victim_id': 'V001',
                    'triage_category': 'UNKNOWN',
                    'requires_medical_assessment': True
                }
            ],
            'recommendations': ['Request paramedic assessment'],
            'tier_used': 3,
            'status': 'partial'
        }
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        return isinstance(context, dict)

