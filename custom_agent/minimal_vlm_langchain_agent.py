"""Minimal VLM agent using LangChain chat models directly instead of VLM class."""
import random
import logging
import numpy as np
import time
from collections import deque

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent

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


@AgentRegistry.register("minimal_vlm_langchain")
class MinimalVLMLangChainAgent(BaseLangChainAgent):
    """Minimal VLM agent using LangChain chat models directly."""
    
    def __init__(self, backend: str = "github_models", model_name: str = "gpt-4o-mini", 
                 temperature: float = 0, **kwargs):
        """
        Initialize minimal VLM agent with LangChain backend.
        
        Args:
            backend: Backend type (github_models, openai, azure_openai, ollama, groq)
            model_name: Name of the model
            temperature: Temperature for generation
            **kwargs: Additional arguments
        """
        super().__init__(backend=backend, model_name=model_name, temperature=temperature, **kwargs)
        self.recent_actions = deque(maxlen=10)
        logger.info(f"Initialized MinimalVLMLangChainAgent with {backend}/{model_name}")
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> list:
        """
        Choose action using LangChain-based VLM decision logic.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Raw screenshot frame as numpy array
            **kwargs: Additional arguments (not used currently)
            
        Returns:
            List of action strings
        """
        from utils.state_formatter import format_state_summary
        
        game_data = state_data.get('game', {})
        
        # Build compact prompt
        state_summary = format_state_summary(state_data)
        recent_str = ", ".join(list(self.recent_actions)) if self.recent_actions else "(none)"
        
        prompt = action_prompt.format(state_summary=state_summary, recent_str=recent_str)
        logger.info(f"State summary: {state_summary}")
        logger.info(f"Recent actions: {recent_str}")
        
        start_time = time.time()
        image_base64 = self._prepare_image(frame)
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        try:
            response = self._call_completion(messages)
            result = self._extract_content(response)
            duration = time.time() - start_time
            self._log_llm_interaction(prompt, result, duration, module_name="ACTION")

            action_response = result.split("\n")[-1].strip().upper()
        except Exception as e:
            duration = time.time() - start_time

            self._log_llm_error(prompt, str(e), duration, module_name="ACTION")
            logger.error(f"API error: {e}")
            action_response = ""
        
        # Parse response into valid buttons
        valid_buttons = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']
        actions = [btn.strip() for btn in action_response.split(',') if btn.strip() in valid_buttons]
        
        # Fallbacks
        if not actions:
            if game_data.get('in_battle', False):
                actions = ['A']
            else:
                actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]
        
        # Update recent actions
        if isinstance(actions, list):
            for action in actions:
                self.recent_actions.append(action)
        else:
            self.recent_actions.append(actions)
        return actions
