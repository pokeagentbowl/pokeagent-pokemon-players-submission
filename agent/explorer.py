"""
ReAct Agent for Pokemon Emerald
================================

Implements a ReAct (Reasoning and Acting) agent that follows the pattern:
Thought -> Action -> Observation -> Thought -> ...

This agent explicitly reasons about the game state before taking actions,
making the decision process more interpretable and debuggable.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from agent.system_prompt import system_prompt


class ActionType(Enum):
    """Possible action types in the ReAct framework."""
    PRESS_BUTTON = "press_button"

@dataclass
class Thought:
    """Represents a reasoning step."""
    content: str


@dataclass
class Action:
    """Represents an action to take."""
    type: ActionType
    parameters: Dict[str, Any]
    justification: str = ""

@dataclass
class Observation:
    """Represents an observation from the environment."""
    content: str
    source: str  # game_state, memory, perception
    timestamp: float = 0.0


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    step_number: int = 0


class ExplorerAgent:
    """
    ReAct Agent that explicitly reasons before acting.
    
    This agent maintains a history of thoughts, actions, and observations
    to make informed decisions about what to do next in the game.
    """
    
    def __init__(
        self,
        vlm_client: Optional[VLM] = None,
        max_history_length: int = 20,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            vlm_client: Vision-language model client for reasoning
            max_history_length: Maximum number of steps to keep in history
            verbose: Whether to print detailed reasoning
        """
        self.vlm_client = vlm_client or VLM()
        self.max_history_length = max_history_length
        self.verbose = verbose
        
        self.history: List[ReActStep] = []
        self.current_step = 0
        self.current_plan: List[str] = []
        self.memory: Dict[str, Any] = {}
        
        self.llm_logger = LLMLogger()
        self.system_prompt = system_prompt
        
    def think(self, state: Dict[str, Any], screenshot: Any = None) -> Thought:
        """
        Generate a thought about the current situation.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            A Thought object with reasoning about the situation
        """
        prompt = self._build_thought_prompt(state, screenshot)
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, prompt, "react")
        else:
            response = self.vlm_client.get_text_query(prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_think",
            prompt=prompt,
            response=response
        )
        
        # Parse the thought from the response
        thought = self._parse_thought(response)
        
        if self.verbose:
            print(f"==> THOUGHT: {thought.content}")
            
        return thought
    
    def act(self, thought: Thought, state: Dict[str, Any]) -> Action:
        """
        Decide on an action based on a thought and current state.
        
        Args:
            thought: The current reasoning
            state: Current game state
            
        Returns:
            An Action object describing what to do
        """
        prompt = self._build_action_prompt(thought, state)
        
        response = self.vlm_client.get_text_query(prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_act",
            prompt=prompt,
            response=response
        )
        
        # Parse the action from the response
        action = self._parse_action(response)
        
        if self.verbose:
            print(f">> ACTION: {action.type.value} - {action.parameters}")
            
        return action
    
    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one complete ReAct step.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            Button press command for the game
        """
        print("###")
        print("Current State:", state)
        print("###")
        self.current_step += 1
        
        # Think about the situation
        thought = self.think(state, screenshot)
        
        # Decide on an action
        action = self.act(thought, state)
        
        # Create step record (observation will be added after action execution)
        step = ReActStep(
            thought=thought,
            action=action,
            step_number=self.current_step
        )
        
        # Add to history
        self._add_to_history(step)
        
        # Convert action to button press
        return self._action_to_button(action)
    
    def _build_thought_prompt(self, state: Dict[str, Any], screenshot: Any) -> str:
        """Build prompt for generating thoughts."""
        recent_history = self._get_recent_history_summary()
        
        prompt = f"""
You are playing Pokemon Emerald. Analyze the current situation and think about what's happening.

CURRENT STATE:
{json.dumps(state, indent=2)}

RECENT HISTORY:
{recent_history}

Based on the current state and what has happened recently, provide your reasoning about:
1. What is currently happening in the game?
2. What challenges or opportunities do you see?
3. What should be the immediate priority?

Respond with your thought in this format:
THOUGHT: [Your detailed reasoning]
"""
        return prompt
    
    def _build_action_prompt(self, thought: Thought, state: Dict[str, Any]) -> str:
        """Build prompt for deciding on actions."""
        prompt = f"""
Based on your reasoning, decide on the next action to take.

YOUR THOUGHT:
{thought.content}

CURRENT STATE:
Player Position: {state.get('player_position', 'unknown')}
Current Map: {state.get('current_map', 'unknown')}
Battle Active: {state.get('battle_active', False)}

Available actions:
- press_button: Press a game button (A, B, UP, DOWN, LEFT, RIGHT)

Respond with your action in this format:
ACTION_TYPE: [press_button/remember/plan/wait]
PARAMETERS: [JSON object with action parameters]
JUSTIFICATION: [Brief explanation of why this action]

Example:
ACTION_TYPE: press_button
PARAMETERS: {{"button": "A"}}
JUSTIFICATION: Interact with the NPC in front of us
"""
        return prompt
    
    def _parse_thought(self, response: str) -> Thought:
        """Parse a thought from LLM response."""
        lines = response.strip().split('\n')
        
        thought_content = response
        
        for line in lines:
            if line.startswith("THOUGHT:"):
                thought_content = line.split(":", 1)[1].strip()
        
        return Thought(
            content=thought_content
        )
    
    def _parse_action(self, response: str) -> Action:
        """Parse an action from LLM response."""
        lines = response.strip().split('\n')
        
        action_type = ActionType.WAIT
        parameters = {}
        justification = ""
        
        for line in lines:
            line = line.strip()  # Strip whitespace from each line
            if line.startswith("ACTION_TYPE:"):
                type_str = line.split(":", 1)[1].strip()
                try:
                    action_type = ActionType(type_str)
                except:
                    action_type = ActionType.WAIT
            elif line.startswith("PARAMETERS:"):
                param_str = line.split(":", 1)[1].strip()
                try:
                    parameters = json.loads(param_str)
                except:
                    parameters = {}
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
        
        return Action(
            type=action_type,
            parameters=parameters,
            justification=justification
        )
    
    def _action_to_button(self, action: Action) -> str:
        """Convert an Action to a button press command."""
        if action.type == ActionType.PRESS_BUTTON:
            return action.parameters.get("button", "NONE")
    
    def _add_to_history(self, step: ReActStep):
        """Add a step to history, maintaining max length."""
        self.history.append(step)
        
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
    
    def _get_recent_history_summary(self) -> str:
        """Get a summary of recent history for context."""
        if not self.history:
            return "No previous actions"
        
        recent = self.history[-5:]  # Last 5 steps
        summary = []
        
        for step in recent:
            if step.thought:
                summary.append(f"Step {step.step_number}: Thought: {step.thought.content[:100]}...")
            if step.action:
                summary.append(f"  Action: {step.action.type.value}")
            if step.observation:
                summary.append(f"  Observed: {step.observation.content[:100]}...")
        
        return "\n".join(summary)

# Convenience function for integration with existing codebase
def create_explorer_agent(**kwargs) -> ExplorerAgent:
    """Create a ReAct agent with default settings."""
    return ExplorerAgent(**kwargs)