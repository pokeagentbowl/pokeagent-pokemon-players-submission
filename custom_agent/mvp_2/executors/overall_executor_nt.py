"""
Overall executor NT - wraps overall_agent_nt as an executor.

This executor delegates to overall_agent_nt which handles:
- Objective planning from JSON files
- Dialogue detection and handling
- Entity-based navigation
- Level grinding
- Menu/battle VLM fallback

Completes when player reaches "LITTLEROOT TOWN BRENDANS HOUSE 2F" map, handing over to standard executor selection.
"""

from typing import List, Optional, TYPE_CHECKING
import logging
import numpy as np
from PIL import Image

from custom_agent.mvp_2.executors.base_executor import BaseExecutor, ExecutorResult
from custom_agent.overall_agent_nt import OverallAgentNT
from custom_utils.detectors import detect_dialogue

if TYPE_CHECKING:
    from custom_agent.mvp_2.modules.perception import PerceptionResult

logger = logging.getLogger(__name__)


class OverallExecutorNT(BaseExecutor):
    """
    Overall executor NT - wraps overall_agent_nt for early game objectives.
    
    This executor manages the initial game flow (van, house, clock, etc.) using
    overall_agent_nt's file-scaffolded planning approach.
    
    Completes when player reaches "LITTLEROOT TOWN BRENDANS HOUSE 2F" map.
    """
    
    def __init__(
        self,
        backend: str = "github_models",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        **kwargs
    ):
        """
        Initialize OverallExecutorNT.
        
        Args:
            backend: LLM backend type
            model_name: Model name
            temperature: Generation temperature
            **kwargs: Additional arguments passed to OverallAgentNT
        """
        super().__init__()
        
        # Initialize the overall agent
        self.overall_agent = OverallAgentNT(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        
        # Track if we've reached LITTLEROOT TOWN
        self.internal_state = {
            'reached_littleroot': False,
            'last_player_map': None
        }
        
        logger.info("Initialized OverallExecutorNT")
    
    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: str
    ) -> ExecutorResult:
        """
        Execute step by delegating to overall_agent_nt.
        
        Args:
            perception: Current perception result
            state_data: Game state data
            goal: Goal description (ignored, overall_agent_nt uses JSON objectives)
        
        Returns:
            ExecutorResult with actions and completion status
        """
        # Get frame from perception or state_data
        frame = perception.frame if hasattr(perception, 'frame') and perception.frame is not None else state_data.get('frame')
        
        # Convert PIL Image to numpy array if needed
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        
        if frame is None:
            logger.error("No frame available for overall_agent_nt")
            return ExecutorResult(
                actions=[],
                status='failed',
                summary="No frame available"
            )
        
        # Get player map for completion check
        player_data = state_data.get('player', {})
        player_map = player_data.get('location', 'unknown')
        self.internal_state['last_player_map'] = player_map
        
        # Check if we've reached LITTLEROOT TOWN BRENDANS HOUSE 2F
        if player_map.upper() == "LITTLEROOT TOWN BRENDANS HOUSE 2F":
            logger.info("Player reached LITTLEROOT TOWN BRENDANS HOUSE 2F - OverallExecutorNT completing")
            self.internal_state['reached_littleroot'] = True
            return ExecutorResult(
                actions=[],
                status='completed',
                summary="Reached LITTLEROOT TOWN BRENDANS HOUSE 2F - handing over to standard executors"
            )
        
        # Delegate to overall_agent_nt
        try:
            result = self.overall_agent.choose_action(state_data, frame)
            if isinstance(result, list):
                actions = result
                reasoning = None
            else:
                actions = result.get('action', [])
                reasoning = result.get('reasoning', None)
            
            return ExecutorResult(
                actions=actions,
                status='in_progress',
                summary=None,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in overall_agent_nt: {e}")
            return ExecutorResult(
                actions=[],
                status='failed',
                summary=f"Error in overall_agent_nt: {str(e)}"
            )
    
    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        Check if OverallExecutorNT is still valid.
        
        Invalid if player has reached "LITTLEROOT TOWN".
        
        Args:
            state_data: Game state data
            perception: Current perception result
        
        Returns:
            bool: True if still valid (not at LITTLEROOT TOWN), False otherwise
        """
        player_data = state_data.get('player', {})
        player_map = player_data.get('location', 'unknown')
        
        # Invalid if we've reached LITTLEROOT TOWN BRENDANS HOUSE 2F
        if player_map.upper() == "LITTLEROOT TOWN BRENDANS HOUSE 2F":
            logger.info("OverallExecutorNT is invalid - player reached LITTLEROOT TOWN BRENDANS HOUSE 2F")
            return False
        
        return True
    
    def reset(self):
        """Reset OverallExecutorNT to initial state."""
        self.internal_state = {
            'reached_littleroot': False,
            'last_player_map': None
        }
        # Note: We don't reset the overall_agent itself to preserve objectives state
        logger.info("Reset OverallExecutorNT state")
