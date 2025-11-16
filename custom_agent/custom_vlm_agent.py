"""Custom VLM agent using LangChain structured output for action decisions."""
import random
import logging
import numpy as np
import time
from collections import deque
from typing import List, Literal
from pydantic import BaseModel, Field

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent

logger = logging.getLogger(__name__)

valid_buttons = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']

action_prompt_old = """
You are the agent playing Pokemon Emerald. 
Your high level goal is to complete the game as quickly as possible.

## Info
STATE SUMMARY: {state_summary}
RECENT ACTIONS: {recent_str}

COMPREHENSIVE STATE INFORMATION: {state_context}

## Task
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
What are the key things in the image? What are the key things in the state summary?
Then, in a new line,

Return ONLY the button name(s) as a comma-separated list, nothing else.
Maximum 10 actions in sequence. 
"""

action_prompt = """
You are the agent playing Pokemon Emerald. 
Your high level goal is to complete the game as quickly as possible.

## Info
STATE SUMMARY: {state_summary}
RECENT ACTIONS: {recent_str}

## Task
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
What are the key things in the image? What are the key things in the state summary?
Then, in a new line,

Return ONLY the button name(s) as a comma-separated list, nothing else.
Maximum 10 actions in sequence. 
"""


class ActionDecision(BaseModel):
    """Structured output for action decisions."""
    reasoning: str = Field(description="Step by step analysis and plan")
    button_names: List[Literal['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']] = Field(description="List of button names to press")
    # note need python 3.11+ to use Literal[*valid_buttons] unwrapping


@AgentRegistry.register("custom_vlm")
class CustomVLMAgent(BaseLangChainAgent):
    """Custom VLM agent using LangChain structured output."""
    
    def __init__(self, backend: str = "github_models", model_name: str = "gpt-4o-mini", 
                 temperature: float = 0, **kwargs):
        """
        Initialize custom VLM agent with LangChain backend and structured output.
        
        Args:
            backend: Backend type (github_models, openai, azure_openai, ollama, groq)
            model_name: Name of the model
            temperature: Temperature for generation
            **kwargs: Additional arguments
        """
        super().__init__(backend=backend, model_name=model_name, temperature=temperature, **kwargs)
        self.recent_actions = deque(maxlen=10)
        
        # Create structured output model
        self.structured_vlm = self.vlm.with_structured_output(ActionDecision)
        logger.info(f"Initialized CustomVLMAgent with {backend}/{model_name} and structured output")
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> list:
        """
        Choose action using LangChain-based VLM with structured output.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Raw screenshot frame as numpy array
            **kwargs: Additional arguments (not used currently)
            
        Returns:
            List of action strings
        """
        from utils.state_formatter import format_state_summary, format_state_for_llm
        
        game_data: dict = state_data.get('game', {})
        
        # Build prompt with comprehensive state context
        state_summary = format_state_summary(state_data)
        state_context = format_state_for_llm(state_data)
        recent_str = ", ".join(list(self.recent_actions)) if self.recent_actions else "(none)"
        
        prompt = action_prompt.format(
            state_summary=state_summary, 
            recent_str=recent_str,
            state_context=state_context
        )
        logger.info(f"State context: {state_context}")
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
            # Use structured output to get ActionDecision object
            response: ActionDecision = self.structured_vlm.invoke(messages)
            duration = time.time() - start_time
            
            # Log the interaction
            self._log_llm_interaction(
                prompt, 
                f"Reasoning: {response.reasoning}\nButtons: {response.button_names}", 
                duration, 
                module_name="ACTION"
            )
            
            logger.info(f"VLM reasoning: {response.reasoning}")
            logger.info(f"VLM button decisions: {response.button_names}")
            
            actions = response.button_names[:10]  # no need checks due to validation in the model
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_llm_error(prompt, str(e), duration, module_name="ACTION")
            logger.error(f"API error: {e}")
            actions = []
        
        # Fallbacks
        if not actions:
            if game_data.get('in_battle', False):
                actions = ['A']
            else:
                actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]
        
        # Update recent actions
        for action in actions:
            self.recent_actions.append(action)
        
        return actions
