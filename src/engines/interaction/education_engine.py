"""
Education Engine - Homework Help and Learning

Helps children with homework, explains concepts, provides tutoring.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from typing import Dict, Any, List
from src.engines.base_engine import BaseEngine, EngineResponse, EngineStatus, EngineTier

logger = logging.getLogger(__name__)


class EducationEngine(BaseEngine):
    """Educational assistance engine"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.openai_client = None
        
        if config and config.get("openai_api_key"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=config["openai_api_key"])
            except:
                pass
    
    def execute(self, entities: Dict[str, Any], context: Dict[str, Any]) -> EngineResponse:
        subject = entities.get("subject", "general")
        question = entities.get("question", "")
        
        if self.openai_client and question:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"You are a patient teacher helping a child with {subject}. Explain in simple, clear terms with examples."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=300
                )
                explanation = response.choices[0].message.content
                
                return EngineResponse(
                    status=EngineStatus.SUCCESS,
                    message=explanation,
                    data={"subject": subject, "llm_used": True},
                    tier_used=EngineTier.TIER_1,
                    execution_time=0.5,
                    confidence=0.95
                )
            except:
                pass
        
        # Fallback
        explanations = {
            "math": "Let's work through this step by step! What problem are you solving?",
            "science": "Great question! Let me explain that in a simple way...",
            "reading": "Let's sound it out together! What word are you working on?",
            "history": "That's an interesting topic! Let me tell you about it...",
        }
        
        message = explanations.get(subject, "I can help with that! Tell me more about what you're working on.")
        
        return EngineResponse(
            status=EngineStatus.SUCCESS,
            message=message,
            data={"subject": subject},
            tier_used=EngineTier.TIER_3,
            execution_time=0.01,
            confidence=0.80
        )
    
    def get_capabilities(self) -> List[str]:
        return ["homework_help", "explain_concept", "tutoring", "quiz", "learning_assistant"]
