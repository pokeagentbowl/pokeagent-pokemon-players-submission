"""
LangChain VLM helper class for making VLM calls with logging, structured output, and Langfuse tracing.
This class can be used independently by reasoning modules or composed into agents.
"""
import logging
import base64
import time
import numpy as np

from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import Union, List, Optional, Type, TypeVar

from langchain_core.messages import BaseMessage

from utils.langfuse_session import initialize_langfuse_session
from utils.vlm import build_langfuse_metadata, get_langfuse_session_id, retry_with_exponential_backoff
from utils.llm_logger import log_llm_interaction, log_llm_error
from custom_utils.llm import create_llm

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LangChainVLM:
    """Helper class for LangChain VLM operations with logging and structured output."""
    
    def __init__(self, backend: str = "github_models", model_name: str = "gpt-4o-mini", 
                 temperature: float = 0, use_langfuse: bool = True, **kwargs):
        """
        Initialize LangChain VLM helper with LangChain chat model backend.
        
        Args:
            backend: LLM backend type (github_models, openai, azure_openai, ollama, groq)
            model_name: Name of the model
            temperature: Temperature for generation
            debug_state: Enable LangChain debug mode (not used here, for compatibility)
            use_langfuse: Enable Langfuse tracing (default: True)
            **kwargs: Additional arguments passed to create_llm
        """
        
        # Initialize Langfuse handler if enabled
        self.use_langfuse = use_langfuse
        self.langfuse_handler = None
        self.langfuse_session_id = None

        if self.use_langfuse:
            try:
                self.langfuse_session_id = initialize_langfuse_session()
            except Exception as e:
                logger.warning(f"Failed to compute Langfuse session ID: {e}. Continuing without Langfuse.")
                self.use_langfuse = False

        if self.use_langfuse:
            try:
                # Lazy import to avoid issues with module loading
                from langfuse.langchain import CallbackHandler
                self.langfuse_handler = CallbackHandler()
                self.langfuse_session_id = self.langfuse_session_id or get_langfuse_session_id()
                logger.info("Langfuse tracing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse handler: {e}. Continuing without Langfuse.")
                self.langfuse_handler = None
                self.use_langfuse = False
        else:
            self.langfuse_session_id = get_langfuse_session_id()
        
        # Create the LangChain chat model
        self.backend = backend
        self.model_name = model_name
        self.vlm = create_llm(
            llm_type=backend,
            model_name=model_name,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"Initialized LangChainVLM with backend={backend}, model={model_name}")
    
    def _build_langfuse_config(self, metadata: dict = None) -> dict:
        """
        Build LangChain config with Langfuse callbacks and metadata.
        
        Args:
            metadata: Optional metadata dict to include in Langfuse trace
            
        Returns:
            dict: Config dict with callbacks and metadata, or empty dict if Langfuse disabled
        """
        config = {}
        if not (self.use_langfuse and self.langfuse_handler):
            return config

        config["callbacks"] = [self.langfuse_handler]

        metadata_payload = dict(metadata) if metadata else {}
        game_state = None
        if isinstance(metadata_payload.get("game_state"), dict):
            game_state = metadata_payload.pop("game_state")

        combined_metadata = build_langfuse_metadata(game_state=game_state, extra_metadata=metadata_payload)
        if self.langfuse_session_id and "langfuse_session_id" not in combined_metadata:
            combined_metadata["langfuse_session_id"] = self.langfuse_session_id

        if combined_metadata:
            config["metadata"] = combined_metadata

        return config
    
    # helpers
    @retry_with_exponential_backoff
    def _call_completion(self, messages: List[BaseMessage], metadata: dict = None, vlm=None) -> BaseMessage:
        """
        Calls the LangChain chat model with exponential backoff.
        
        Args:
            messages: List of messages to send
            metadata: Optional metadata to include in Langfuse trace
            vlm: Optional VLM to use. If None, uses self.vlm
        """
        # Use provided vlm or fall back to self.vlm
        model = vlm if vlm is not None else self.vlm
        
        # Build Langfuse config (includes session ID from env var)
        config = self._build_langfuse_config(metadata=metadata)
        
        return model.invoke(messages, config=config)

    def _extract_content(self, response: BaseMessage) -> str:
        """Extract text content from the LangChain response."""
        return response.content
    
    # Image preparation
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]):
        """Convert image to base64 for VLM input."""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return image_base64

    # Logging methods
    def _log_llm_interaction(self, text: str, result: str, duration: float, module_name: str = "Unknown", metadata: Optional[dict] = None):
        """Log successful LLM interaction."""
        log_metadata = {"backend": self.backend}
        if metadata:
            log_metadata.update(metadata)
        log_llm_interaction(
            interaction_type=f"{self.backend}_{module_name}",
            prompt=text,
            response=result,
            duration=duration,
            metadata=log_metadata,
            model_info={"model": self.model_name, "backend": self.backend}
        )

    def _log_llm_error(self, text: str, error: str, duration: float, module_name: str = "Unknown", metadata: Optional[dict] = None):
        """Log LLM error."""
        log_metadata = {"backend": self.backend, "duration": duration}
        if metadata:
            log_metadata.update(metadata)
        log_metadata.setdefault("model", self.model_name)
        log_llm_error(
            interaction_type=f"{self.backend}_{module_name}",
            prompt=text,
            error=error,
            duration=duration,
            metadata=log_metadata
        )

    # Main VLM call method
    def call_vlm(
        self,
        prompt: str,
        image: Optional[Union[Image.Image, np.ndarray]] = None,
        module_name: str = "Unknown",
        structured_output_model: Optional[Type[T]] = None,
        metadata: Optional[dict] = None
    ) -> Union[str, T]:
        """
        Call VLM with optional image and automatic logging.
        Supports both regular text responses and structured Pydantic outputs.
        
        Args:
            prompt: Text prompt
            image: Optional image (PIL Image or numpy array)
            module_name: Name for logging purposes (e.g., "ACTION", "NAVIGATION")
            structured_output_model: Optional Pydantic model class for structured output
            metadata: Optional metadata dict to include in Langfuse trace
        
        Returns:
            str if structured_output_model is None, otherwise instance of the Pydantic model
            
        Raises:
            Exception: If VLM call fails (after logging error)
        """
        start_time = time.time()
        call_metadata = {
            "backend": self.backend,
            "module": module_name,
            "model": self.model_name,
            "has_image": image is not None,
        }
        if metadata:
            call_metadata.update(metadata)
        
        try:
            # Build messages with or without image
            if image is not None:
                image_base64 = self._prepare_image(image)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }]
            else:
                messages = [{
                    "role": "user",
                    "content": prompt
                }]
            
            # Choose invocation method based on structured output requirement
            if structured_output_model is not None:
                structured_vlm = self.vlm.with_structured_output(structured_output_model)
                # Use _call_completion with the structured vlm
                response = self._call_completion(messages, metadata=call_metadata, vlm=structured_vlm)
                # Format Pydantic model for logging
                result_str = response.model_dump_json(indent=4)
            else:
                response = self._call_completion(messages, metadata=call_metadata)
                result_str = self._extract_content(response)
            
            # Log success
            duration = time.time() - start_time
            self._log_llm_interaction(prompt, result_str, duration, module_name=module_name, metadata=call_metadata)
            
            # Return appropriate type
            return response if structured_output_model is not None else result_str
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_llm_error(prompt, str(e), duration, module_name=module_name, metadata=call_metadata)
            logger.error(f"VLM call failed ({module_name}): {e}")
            raise
