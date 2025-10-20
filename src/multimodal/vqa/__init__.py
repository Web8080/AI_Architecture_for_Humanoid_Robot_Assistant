"""
Visual Question Answering Package

Implements VQA system for answering questions about visual scenes
using both vision and language understanding.

Author: Victor Ibhafidon
Date: October 2025
"""

from .visual_qa import VisualQuestionAnswering, VQAResult, VQAQuestion, QuestionType

__all__ = [
    'VisualQuestionAnswering',
    'VQAResult',
    'VQAQuestion',
    'QuestionType'
]
