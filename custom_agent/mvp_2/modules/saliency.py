"""Saliency detector - coordinates executor validity checks."""

from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from custom_agent.mvp_2.modules.perception import PerceptionResult
    from custom_agent.mvp_2.executors.base_executor import ExecutorProtocol


class SaliencyResult(BaseModel):
    """Result of saliency detection."""
    executor_valid: bool
    reason: str


class SaliencyDetector:
    """
    Coordinates executor validity checks.
    
    Does NOT contain game state checks directly - delegates to executors.
    """
    
    def check_validity(
        self,
        current_executor: Optional['ExecutorProtocol'],
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> SaliencyResult:
        """
        Check if current executor is still valid.
        
        Args:
            current_executor: Current executor (or None)
            state_data: Game state data
            perception: Perception result
            
        Returns:
            SaliencyResult with validity status and reason
        """
        if current_executor is None:
            return SaliencyResult(
                executor_valid=False,
                reason="No current executor"
            )
        
        # Delegate to executor's own validity check
        is_valid = current_executor.is_still_valid(state_data, perception)
        
        if not is_valid:
            return SaliencyResult(
                executor_valid=False,
                reason=f"{current_executor.__class__.__name__} became invalid"
            )
        
        return SaliencyResult(
            executor_valid=True,
            reason="Executor still valid"
        )

