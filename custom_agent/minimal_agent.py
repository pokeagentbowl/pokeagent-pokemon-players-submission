import random
import logging
import numpy as np

from custom_agent.base_agent import BaseAgent, AgentRegistry

logger = logging.getLogger(__name__)

def simple_action_decision(state_data: dict, frame: np.ndarray, step: int) -> list[str]:
    """
    Placeholder function that takes in emulator inputs and outputs a sequence of actions.
    
    Args:
        state_data: Raw state data from the emulator
        frame: Raw screenshot frame as numpy array
        step: Current step number
        
    Returns:
        list[str]: List of 5 actions to press (e.g., 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B')
    """
    # This is a simple placeholder that demonstrates access to raw emulator data
    # In a real implementation, this would contain the actual decision logic
    
    # Access raw emulator data without processing
    game_data = state_data.get('game', {})

    # Log the raw state data structure
    logger.info(f"[STEP-{step}] Raw state keys available: {list(state_data.keys())}")
    if game_data:
        logger.info(f"[STEP-{step}] Game state keys: {list(game_data.keys())}")
    if 'player' in state_data:
        logger.info(f"[STEP-{step}] Player state keys: {list(state_data['player'].keys())}")
    if 'map' in state_data:
        logger.info(f"[STEP-{step}] Map state keys: {list(state_data['map'].keys())}")
    
    # Simple decision logic based on raw state
    if game_data.get('in_battle', False):
        # In battle - send five 'A' presses
        # logger.info(f"[STEP-{step}] In battle - sending five 'A' presses")
        # return ['A'] * 5
        logger.info(f"[STEP-{step}] In battle - sending one 'A' press")
        return 'A'
    else:
        # Not in battle - choose 5 random actions from allowed set
        allowed_actions = ['A', 'B', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        # actions = [random.choice(allowed_actions) for _ in range(5)]
        # logger.info(f"[STEP-{step}] Overworld - random 5-button sequence: {actions}")
        actions = random.choice(allowed_actions)
        logger.info(f"[STEP-{step}] Overworld - random action: {actions}")
        return actions


@AgentRegistry.register("minimal")
class MinimalAgent(BaseAgent):
    """Minimal agent that makes random action decisions."""
    
    def __init__(self, **kwargs):
        """Initialize minimal agent. No backend needed."""
        pass
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> str:
        """
        Choose action using simple random decision logic.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Raw screenshot frame as numpy array
            **kwargs: Additional arguments (step number, etc.)
            
        Returns:
            Single action string
        """
        step = kwargs.get('step', 0)
        return simple_action_decision(state_data, frame, step)
