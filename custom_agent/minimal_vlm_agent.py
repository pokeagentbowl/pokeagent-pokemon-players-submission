import random
import logging
import numpy as np
from collections import deque

from utils.vlm import VLM
from custom_agent.base_agent import AgentRegistry
from custom_agent.base_vlm_agent import BaseVLMAgent

logger = logging.getLogger(__name__)

action_prompt = """
You are the agent playing Pokemon Emerald with a speedrunning mindset. Make quick, efficient decisions.

STATE SUMMARY: {state_summary}
RECENT ACTIONS: {recent_str}

Based on the comprehensive state information above and the image frame, 
think step by step to analyze the situation and make a plan, 
then decide your next action(s):

Valid buttons: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, L, R
- A: Interact, confirm, attack
- B: Cancel, back, run (with running shoes)
- START: Main menu
- SELECT: Use registered item
- Directional: Move, navigate menus (use movement options above)
- L/R: Cycle through options in some menus

## Output format
First output your step by step analysis and plan. 
Then, in a new line,

Return ONLY the button name(s) as a comma-separated list, nothing else.
Maximum 10 actions in sequence. 
"""

def choose_actions_vlm(state_data: dict, frame: np.ndarray, recent_actions: list[str], vlm: VLM) -> list[str]:
    """
    Decide next action(s) using VLM each step with minimal context.

    Returns a list of valid button strings.
    """
    from utils.state_formatter import format_state_summary
    
    game_data = state_data.get('game', {})
    
    # Build compact prompt
    # if not good enough, can try format_state_for_llm
    state_summary = format_state_summary(state_data)
    recent_str = ", ".join(list(recent_actions)) if recent_actions else "(none)"

    prompt = action_prompt.format(state_summary=state_summary, recent_str=recent_str)
    logger.info(f"State summary: {state_summary}")
    logger.info(f"Recent actions: {recent_str}")
    
    try:
        vlm_response = vlm.get_query(frame, prompt, "ACTION")
        action_response = vlm_response.split("\n")[-1].strip().upper()
    except Exception as e:
        logger.error(f"Error choosing actions via VLM actor: {e}")
        action_response = ""
    
    # Parse response into valid buttons
    valid_buttons = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']
    actions = []

    actions = [btn.strip() for btn in action_response.split(',') if btn.strip() in valid_buttons]
    
    # Fallbacks
    if not actions:
        if game_data.get('in_battle', False):
            actions = ['A']
        else:
            actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]
    
    return actions


@AgentRegistry.register("minimal_vlm")
class MinimalVLMAgent(BaseVLMAgent):
    """Minimal VLM agent that uses VLM backend for decision making."""
    
    def __init__(self, model_name: str = "gpt-5-nano", backend: str = "auto", 
                 base_url: str = None, **kwargs):
        """
        Initialize minimal VLM agent with VLM backend.
        
        Args:
            model_name: Name of the VLM model
            backend: Backend type (auto, openai, openrouter, gemini, openai_langchain)
            base_url: Optional base URL for API
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, backend=backend, base_url=base_url, **kwargs)
        self.recent_actions = deque(maxlen=10)
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> list:
        """
        Choose action using VLM-based decision logic.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Raw screenshot frame as numpy array
            **kwargs: Additional arguments (not used currently)
            
        Returns:
            List of action strings
        """
        actions = choose_actions_vlm(state_data, frame, self.recent_actions, self.vlm)
        # Update recent actions
        if isinstance(actions, list):
            for action in actions:
                self.recent_actions.append(action)
        else:
            self.recent_actions.append(actions)
        return actions

