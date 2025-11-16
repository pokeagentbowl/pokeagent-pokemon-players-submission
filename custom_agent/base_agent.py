"""Base agent class and registry for modular agent architecture."""
from abc import ABC
from typing import Dict, Type, Union
import numpy as np


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> Union[str, list[str], dict]:
        """
        Decide next action(s) based on state and frame.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Game frame as numpy array
            **kwargs: Additional arguments
            
        Returns:
            Action(s) to execute, or dict with 'action' and optional additional data
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either 'choose_action' (v1 legacy) "
            f"or override 'step' (v3 preferred)"
        )

    def step(self, game_state: dict) -> dict:
        """
        This method bridges the interface between v1 (choose_action) and v3 (step)
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning', 'npc_updates', etc.
        """
        frame = np.array(game_state.get('frame'))
        result = self.choose_action(game_state, frame)
        
        # If choose_action returns a dict, use it directly (new format)
        # Otherwise, wrap the result in the legacy format
        if isinstance(result, dict):
            return result
        else:
            return {'action': result}


class AgentRegistry:
    """Registry for agent classes."""
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an agent class."""
        def decorator(agent_class: Type[BaseAgent]):
            cls._agents[name] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def get_agent(cls, name: str) -> Type[BaseAgent]:
        """Get an agent class by name."""
        if name not in cls._agents:
            raise ValueError(f"Agent '{name}' not found. Available agents: {list(cls._agents.keys())}")
        return cls._agents[name]
    
    @classmethod
    def list_agents(cls) -> list:
        """List all registered agent names."""
        return list(cls._agents.keys())
