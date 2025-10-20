"""
AI Agents Package

Implements the complete AI Agent Architecture following the perception → reasoning → 
planning → execution → learning → interaction cycle.

Components:
- ai_agent.py: Main AI Agent orchestrator
- multimodal_fusion.py: Multi-modal data fusion
- reasoning_engine.py: Logic and neural reasoning
- planning_engine.py: Task planning and strategy
- execution_engine.py: Action execution
- learning_engine.py: Continuous learning and adaptation

Author: Victor Ibhafidon
Date: October 2025
"""

from .ai_agent import AIAgent, AgentState, PerceptionResult, ReasoningResult, PlanningResult, ExecutionResult, LearningResult

__all__ = [
    'AIAgent',
    'AgentState', 
    'PerceptionResult',
    'ReasoningResult',
    'PlanningResult',
    'ExecutionResult',
    'LearningResult'
]
