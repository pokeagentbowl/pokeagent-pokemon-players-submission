"""Base agent class for VLM-based agents using the VLM wrapper class."""
import logging
import numpy as np
from abc import abstractmethod

from custom_agent.base_agent import BaseAgent
from utils.vlm import VLM

logger = logging.getLogger(__name__)


class BaseVLMAgent(BaseAgent):
    """Base class for agents using VLM wrapper class."""
    
    def __init__(self, model_name: str = "gpt-5-nano", backend: str = "auto", 
                 base_url: str = None, **kwargs):
        """
        Initialize base VLM agent with VLM backend.
        
        Args:
            model_name: Name of the VLM model
            backend: Backend type (auto, openai, openrouter, gemini, openai_langchain)
            base_url: Optional base URL for API
            **kwargs: Additional arguments
        """
        self.vlm = VLM(model_name=model_name, backend=backend, base_url=base_url)
        logger.info(f"Initialized BaseVLMAgent with backend={backend}, model={model_name}")
    
    @abstractmethod
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs):
        """
        Choose action - must be implemented by subclass.
        
        Args:
            state_data: Raw state data from the emulator
            frame: Raw screenshot frame as numpy array
            **kwargs: Additional arguments
            
        Returns:
            Action or list of actions to execute
        """
        pass
