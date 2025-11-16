"""Base executor protocol and class with shared functionality."""

import numpy as np
from abc import abstractmethod
from typing import Protocol, List, Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from custom_agent.mvp_2.modules.perception import PerceptionResult


class ExecutorResult(BaseModel):
    """Result from executor step."""
    actions: List[str]
    status: Literal['in_progress', 'completed', 'failed']
    summary: Optional[str] = None  # For logging when completed/failed
    reasoning: Optional[str] = None  # For long-term memory (VLM reasoning)


class ExecutorProtocol(Protocol):
    """
    Protocol that all executors must implement.
    
    This ensures consistent interface across all executors.
    """
    
    @abstractmethod
    def execute_step(
        self,
        perception: 'PerceptionResult',  # type: ignore
        state_data: dict,
        goal: any
    ) -> ExecutorResult:
        """
        Execute one step towards the goal.
        
        Should emit small action batch (1-5 actions) for responsiveness.
        
        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Natural language goal from planner (str for general/battle, dict for navigation)
            
        Returns:
            ExecutorResult with:
            - actions: List[str] (small batch)
            - status: 'in_progress' | 'completed' | 'failed'
            - summary: Optional[str] (for logging when completed/failed)
        """
        ...
    
    @abstractmethod
    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'  # type: ignore
    ) -> bool:
        """
        Check if this executor is still valid for the current situation.
        
        Each executor defines its own validity criteria.
        
        Examples:
        - NavigationExecutor: invalid if battle started
        - BattleExecutor: invalid if battle ended
        - GeneralExecutor: context-dependent
        
        Args:
            state_data: Game state data
            perception: Current perception
            
        Returns:
            bool: True if still valid, False if need to replan
        """
        ...
    
    @abstractmethod
    def get_state(self) -> dict:
        """
        Get executor's internal state for suspension/resumption.
        
        Used for goal suspension feature.
        
        Returns:
            dict: Internal state (executor-specific)
        """
        ...
    
    @abstractmethod
    def restore_state(self, state: dict):
        """
        Restore executor's internal state from suspension.

        Used for goal suspension feature.

        Args:
            state: Previously saved state from get_state()
        """
        ...

    @abstractmethod
    def reset(self):
        """
        Reset executor to initial state.

        Called when starting a new goal (not resuming) to clear any stale state.
        """
        ...


class BaseExecutor:
    """
    Base class with common executor functionality.

    Provides default implementations for state management.
    Subclasses must implement execute_step, is_still_valid, and reset.
    """

    def __init__(self):
        self.internal_state = {}

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        raise NotImplementedError("Subclasses must implement execute_step")

    def get_state(self) -> dict:
        """Default state getter - returns copy of internal state."""
        return self.internal_state.copy()

    def restore_state(self, state: dict):
        """Default state restorer - sets internal state."""
        self.internal_state = state.copy()

    def reset(self):
        """Default reset - clears internal state. Subclasses should override if needed."""
        self.internal_state = {}

