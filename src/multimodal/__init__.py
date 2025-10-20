"""
Multimodal AI Package

Implements multimodal fusion, visual grounding, and visual question answering
for humanoid robot applications.

Components:
- grounding/: Visual grounding for referring expressions
- vqa/: Visual Question Answering system
- cross_modal/: Cross-modal retrieval and matching
- vision_language/: Vision-language model integration

Author: Victor Ibhafidon
Date: October 2025
"""

from .grounding.visual_grounding import VisualGrounding, GroundingResult, ReferringExpression, SpatialRelation
from .vqa.visual_qa import VisualQuestionAnswering, VQAResult, VQAQuestion, QuestionType

__all__ = [
    'VisualGrounding',
    'GroundingResult', 
    'ReferringExpression',
    'SpatialRelation',
    'VisualQuestionAnswering',
    'VQAResult',
    'VQAQuestion',
    'QuestionType'
]
