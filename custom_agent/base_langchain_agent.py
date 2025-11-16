"""
Base agent class for LangChain-based agents.
Originally has langchain chat model directly but now uses the LangChainVLM helper object
Users are recommended to compose their own reasoning modules using the LangChainVLM helper object
and use the BaseLangChainAgent as a base class for agents which contain reasoning modules
"""
import logging
import numpy as np

from PIL import Image
from pydantic import BaseModel
from typing import Union, Optional, Type, TypeVar

from langchain_core.globals import set_debug

from custom_agent.base_agent import BaseAgent
from custom_utils.cache_utils import init_langchain_llm_cache
from custom_utils.langchain_vlm import LangChainVLM

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class BaseLangChainAgent(BaseAgent):
    """Base class for agents using LangChain chat models directly."""
    
    def __init__(self, backend: str = "github_models", model_name: str = "gpt-4o-mini", 
                 temperature: float = 0, debug_state: bool = False, use_langfuse: bool = True, **kwargs):
        """
        Initialize base LangChain agent with LangChain chat model backend.
        
        Args:
            backend: LLM backend type (github_models, openai, azure_openai, ollama, groq)
            model_name: Name of the model
            temperature: Temperature for generation
            debug_state: Enable LangChain debug mode
            use_langfuse: Enable Langfuse tracing (default: True)
            **kwargs: Additional arguments passed to create_llm
        """
        # Setup caching and debug before LLM initialization
        set_debug(debug_state)
        init_langchain_llm_cache(db_name="lm_cache.db")

        # Create VLM helper (handles caching, langfuse, and LLM creation)
        self.vlm_helper = LangChainVLM(
            backend=backend,
            model_name=model_name,
            temperature=temperature,
            use_langfuse=use_langfuse,
            **kwargs
        )
        
        # Expose VLM attributes at agent level for backward compatibility
        self.vlm = self.vlm_helper.vlm
        self.backend = self.vlm_helper.backend
        self.model_name = self.vlm_helper.model_name
        
        logger.info(f"Initialized BaseLangChainAgent with backend={backend}, model={model_name}")
    
    # Delegate all VLM methods to vlm_helper for backward compatibility
    def _call_completion(self, *args, **kwargs):
        """Delegate to VLM helper."""
        return self.vlm_helper._call_completion(*args, **kwargs)
    
    def _extract_content(self, *args, **kwargs):
        """Delegate to VLM helper."""
        return self.vlm_helper._extract_content(*args, **kwargs)
    
    def _prepare_image(self, *args, **kwargs):
        """Delegate to VLM helper."""
        return self.vlm_helper._prepare_image(*args, **kwargs)
    
    def _log_llm_interaction(self, *args, **kwargs):
        """Delegate to VLM helper."""
        return self.vlm_helper._log_llm_interaction(*args, **kwargs)
    
    def _log_llm_error(self, *args, **kwargs):
        """Delegate to VLM helper."""
        return self.vlm_helper._log_llm_error(*args, **kwargs)
    
    def call_vlm(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray]] = None,
        module_name: str = "Unknown",
        structured_output_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None
    ) -> Union[str, T]:
        return self.vlm_helper.call_vlm(prompt, image, module_name, structured_output_model, metadata)

    def call_vlm_with_logging(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray]] = None,
        module_name: str = "Unknown",
        structured_output_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None
    ) -> Union[str, T]:
        """Alias for call_vlm (for backward compatibility)."""
        return self.call_vlm(prompt, image, module_name, structured_output_model, metadata)
