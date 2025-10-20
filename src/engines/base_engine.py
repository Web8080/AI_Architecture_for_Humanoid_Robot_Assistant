"""
Base Engine Class for Humanoid Robot Assistant

All engines inherit from this base class, providing a standardized interface
for intent handling, execution, and response generation.

Author: Victor Ibhafidon
Date: October 2025
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """Engine execution status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"


class EngineTier(Enum):
    """Multi-tier fallback levels"""
    TIER_1 = "tier_1"  # Best quality, highest resource
    TIER_2 = "tier_2"  # Medium quality, medium resource
    TIER_3 = "tier_3"  # Basic quality, always works


@dataclass
class EngineResponse:
    """Standardized engine response"""
    status: EngineStatus
    message: str
    data: Dict[str, Any]
    tier_used: EngineTier
    execution_time: float
    confidence: float
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "tier_used": self.tier_used.value,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def is_success(self) -> bool:
        """Check if execution was successful"""
        return self.status in [EngineStatus.SUCCESS, EngineStatus.PARTIAL_SUCCESS]


class BaseEngine(ABC):
    """
    Base class for all robot engines
    
    Each engine must implement:
    - execute(): Main execution logic
    - get_capabilities(): Return engine capabilities
    
    Each engine should follow multi-tier fallback pattern:
    - Tier 1: Best quality (e.g., cloud API, GPU model)
    - Tier 2: Medium quality (e.g., local model, CPU)
    - Tier 3: Basic fallback (e.g., rule-based, always works)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize engine
        
        Args:
            config: Engine-specific configuration
        """
        self.config = config if config else {}
        self.name = self.__class__.__name__
        self.enabled = self.config.get("enabled", True)
        self.tier1_enabled = self.config.get("tier1_enabled", True)
        self.tier2_enabled = self.config.get("tier2_enabled", True)
        self.tier3_enabled = self.config.get("tier3_enabled", True)
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def execute(self, entities: Dict[str, Any], context: Dict[str, Any]) -> EngineResponse:
        """
        Main execution method - must be implemented by each engine
        
        Args:
            entities: Extracted entities from NLP
            context: Session context and memory
            
        Returns:
            EngineResponse: Standardized response
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this engine provides
        
        Returns:
            List of capability strings
        """
        pass
    
    def _execute_with_fallback(self, tier1_fn, tier2_fn, tier3_fn, 
                                entities: Dict[str, Any], 
                                context: Dict[str, Any]) -> EngineResponse:
        """
        Execute with multi-tier fallback
        
        Args:
            tier1_fn: Tier 1 execution function
            tier2_fn: Tier 2 execution function
            tier3_fn: Tier 3 execution function (always works)
            entities: Extracted entities
            context: Session context
            
        Returns:
            EngineResponse from the first successful tier
        """
        start_time = time.time()
        
        # Try Tier 1
        if self.tier1_enabled:
            try:
                logger.debug(f"{self.name}: Attempting Tier 1")
                result = tier1_fn(entities, context)
                execution_time = time.time() - start_time
                
                response = EngineResponse(
                    status=EngineStatus.SUCCESS,
                    message=result.get("message", "Tier 1 execution successful"),
                    data=result.get("data", {}),
                    tier_used=EngineTier.TIER_1,
                    execution_time=execution_time,
                    confidence=result.get("confidence", 0.95)
                )
                
                self._update_metrics(response)
                return response
                
            except Exception as e:
                logger.warning(f"{self.name}: Tier 1 failed: {e}. Falling back to Tier 2.")
        
        # Try Tier 2
        if self.tier2_enabled:
            try:
                logger.debug(f"{self.name}: Attempting Tier 2")
                result = tier2_fn(entities, context)
                execution_time = time.time() - start_time
                
                response = EngineResponse(
                    status=EngineStatus.SUCCESS,
                    message=result.get("message", "Tier 2 execution successful"),
                    data=result.get("data", {}),
                    tier_used=EngineTier.TIER_2,
                    execution_time=execution_time,
                    confidence=result.get("confidence", 0.85)
                )
                
                self._update_metrics(response)
                return response
                
            except Exception as e:
                logger.warning(f"{self.name}: Tier 2 failed: {e}. Falling back to Tier 3.")
        
        # Try Tier 3 (always works)
        try:
            logger.debug(f"{self.name}: Attempting Tier 3 (fallback)")
            result = tier3_fn(entities, context)
            execution_time = time.time() - start_time
            
            response = EngineResponse(
                status=EngineStatus.SUCCESS,
                message=result.get("message", "Tier 3 execution successful"),
                data=result.get("data", {}),
                tier_used=EngineTier.TIER_3,
                execution_time=execution_time,
                confidence=result.get("confidence", 0.70),
                warnings=["Using fallback tier"]
            )
            
            self._update_metrics(response)
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{self.name}: All tiers failed: {e}")
            
            response = EngineResponse(
                status=EngineStatus.FAILURE,
                message=f"All tiers failed for {self.name}",
                data={},
                tier_used=EngineTier.TIER_3,
                execution_time=execution_time,
                confidence=0.0,
                errors=[str(e)]
            )
            
            self._update_metrics(response)
            return response
    
    def _update_metrics(self, response: EngineResponse):
        """Update engine performance metrics"""
        self.execution_count += 1
        self.total_execution_time += response.execution_time
        
        if response.is_success():
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0.0)
        success_rate = (self.success_count / self.execution_count 
                       if self.execution_count > 0 else 0.0)
        
        return {
            "engine_name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "average_execution_time": avg_time,
            "total_execution_time": self.total_execution_time
        }
    
    def validate_entities(self, entities: Dict[str, Any], 
                         required_keys: List[str]) -> bool:
        """
        Validate that required entity keys are present
        
        Args:
            entities: Entity dictionary
            required_keys: List of required keys
            
        Returns:
            True if all required keys present
        """
        for key in required_keys:
            if key not in entities or not entities[key]:
                logger.warning(f"{self.name}: Missing required entity: {key}")
                return False
        return True
    
    def get_entity(self, entities: Dict[str, Any], key: str, 
                   default: Any = None) -> Any:
        """Safely get entity value with default"""
        return entities.get(key, default)
    
    def is_enabled(self) -> bool:
        """Check if engine is enabled"""
        return self.enabled
    
    def enable(self):
        """Enable engine"""
        self.enabled = True
        logger.info(f"{self.name}: Enabled")
    
    def disable(self):
        """Disable engine"""
        self.enabled = False
        logger.info(f"{self.name}: Disabled")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        logger.info(f"{self.name}: Metrics reset")
    
    def __str__(self) -> str:
        return f"{self.name}(enabled={self.enabled})"
    
    def __repr__(self) -> str:
        return self.__str__()

