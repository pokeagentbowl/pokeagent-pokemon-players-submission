import base64
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import requests
from PIL import Image

# Set up module logging
logger = logging.getLogger(__name__)

# Import LLM logger
from utils.llm_logger import log_llm_error, log_llm_interaction

# Langfuse integration
_langfuse_enabled = False
_langfuse_session_id = None
_langfuse_user_id = None


def _initialize_langfuse():
    """Initialize Langfuse tracing if enabled"""
    global _langfuse_enabled, _langfuse_user_id

    _langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "yes", "on")
    if not _langfuse_enabled:
        return

    try:
        # Check for required environment variables
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        _langfuse_user_id = os.getenv("LANGFUSE_USER_ID")

        if not public_key or not secret_key:
            logger.warning("Langfuse enabled but missing PUBLIC_KEY or SECRET_KEY")
            _langfuse_enabled = False
            return

        # Note: Langfuse OpenAI wrapper is initialized in OpenAI/OpenRouter/vLLM backends
        # Note: OpenInference instrumentors for Google GenAI and Vertex AI
        # are initialized per-backend in their respective __init__ methods

        logger.info("Langfuse tracing enabled")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        _langfuse_enabled = False


def _prepare_game_state_metadata(game_state: dict) -> dict:
    """
    Prepare game_state for Langfuse metadata, excluding screenshot.
    
    Args:
        game_state: Complete game state dictionary
        
    Returns:
        Sanitized game state dictionary suitable for Langfuse metadata
    """
    if not game_state:
        return {}
    
    try:
        # Create a deep copy to avoid modifying the original
        import copy
        sanitized = copy.deepcopy(game_state)
        
        # Remove screenshot from visual (it's too large for metadata)
        if "visual" in sanitized and "screenshot" in sanitized["visual"]:
            del sanitized["visual"]["screenshot"]
        
        # Remove screenshot_base64 if it exists
        if "visual" in sanitized and "screenshot_base64" in sanitized["visual"]:
            del sanitized["visual"]["screenshot_base64"]
        
        # Keep: game_state, visual (without screenshot), audio, progress
        # Structure: {"game": {...}, "player": {...}, "map": {...}, "visual": {...}}
        
        return sanitized
    except Exception as e:
        logger.warning(f"Failed to prepare game_state metadata: {e}")
        return {}


def _get_langfuse_metadata(game_state: dict = None):
    """
    Get Langfuse metadata for trace attributes (session_id, user_id, game_state).
    
    Args:
        game_state: Optional game state dictionary to include in metadata
        
    Returns:
        Dictionary with Langfuse metadata
    """
    metadata = {}
    if _langfuse_session_id:
        metadata["langfuse_session_id"] = _langfuse_session_id
    if _langfuse_user_id:
        metadata["langfuse_user_id"] = _langfuse_user_id
    
    # Add game_state if provided
    if game_state:
        game_state_metadata = _prepare_game_state_metadata(game_state)
        if game_state_metadata:
            metadata["game_state"] = game_state_metadata
    
    return metadata


def build_langfuse_metadata(game_state: dict | None = None, extra_metadata: dict | None = None) -> dict:
    """Compose Langfuse metadata with optional extras for downstream callers."""
    metadata = _get_langfuse_metadata(game_state) or {}
    if extra_metadata:
        for key, value in extra_metadata.items():
            if value is not None:
                metadata[key] = value
    return metadata


def set_langfuse_session_id(session_id: str):
    """Set the Langfuse session ID for tracing"""
    global _langfuse_session_id
    _langfuse_session_id = session_id
    logger.info(f"Langfuse session ID set to: {session_id}")


def get_langfuse_session_id() -> str | None:
    """Get the current Langfuse session ID"""
    return _langfuse_session_id


# Initialize Langfuse on module import
_initialize_langfuse()


# Define the retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increase the delay with exponential factor and random jitter
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper

class VLMBackend(ABC):
    """Abstract base class for VLM backends"""

    @abstractmethod
    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """
        Process an image and text prompt.
        
        Args:
            img: Image to process
            text: Text prompt
            module_name: Name of the calling module for tracking
            game_state: Optional game state dictionary to include in metadata
        """
        pass

    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """
        Process a text-only prompt.
        
        Args:
            text: Text prompt
            module_name: Name of the calling module for tracking
            game_state: Optional game state dictionary to include in metadata
        """
        pass

class OpenAIBackend(VLMBackend):
    """OpenAI API backend with Langfuse tracing"""

    def __init__(self, model_name: str, **kwargs):
        try:
            import openai
            # Use Langfuse-wrapped OpenAI client if Langfuse is enabled
            if _langfuse_enabled:
                from langfuse.openai import OpenAI
                logger.info("Using Langfuse-wrapped OpenAI client for automatic tracing")
            else:
                from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Error: OpenAI API key is missing! Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.errors = (openai.RateLimitError,)

    @retry_with_exponential_backoff
    def _call_completion(self, messages, **kwargs):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt using OpenAI API"""
        start_time = time.time()

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

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]

        try:
            # Build metadata with Langfuse session_id, user_id, and game_state
            metadata = _get_langfuse_metadata(game_state)
            metadata.update({
                "backend": "openai",
                "module": module_name,
                "has_image": True,
                "langfuse_session_id": _langfuse_session_id,
            })
            
            # Langfuse wrapper automatically traces this call with metadata
            response = self._call_completion(messages, name=f"openai_{module_name}_image", metadata=metadata)
            result = response.choices[0].message.content
            end_time = time.time()
            duration = end_time - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            # Log the interaction (Langfuse wrapper handles tracing automatically)
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )

            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": True}
            )
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt using OpenAI API"""
        start_time = time.time()

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]

        try:
            # Build metadata with Langfuse session_id, user_id, and game_state
            metadata = _get_langfuse_metadata(game_state)
            metadata.update({
                "backend": "openai",
                "module": module_name,
                "has_image": False,
                "langfuse_session_id": _langfuse_session_id,
            })
            
            # Langfuse wrapper automatically traces this call with metadata
            response = self._call_completion(messages, name=f"openai_{module_name}_text", metadata=metadata)
            result = response.choices[0].message.content
            end_time = time.time()
            duration = end_time - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            # Log the interaction (Langfuse wrapper handles tracing automatically)
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )

            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": False}
            )
            logger.error(f"OpenAI API error: {e}")
            raise

class OpenRouterBackend(VLMBackend):
    """OpenRouter API backend with Langfuse tracing"""

    def __init__(self, model_name: str, **kwargs):
        try:
            # Use Langfuse-wrapped OpenAI client if Langfuse is enabled
            if _langfuse_enabled:
                from langfuse.openai import OpenAI
                logger.info("Using Langfuse-wrapped OpenAI client for OpenRouter tracing")
            else:
                from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")

        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("Error: OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    @retry_with_exponential_backoff
    def _call_completion(self, messages, **kwargs):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt using OpenRouter API"""
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

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]

        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        # Build metadata with Langfuse session_id, user_id, and game_state
        metadata = _get_langfuse_metadata(game_state)
        metadata.update({
            "backend": "openrouter",
            "module": module_name,
            "has_image": True,
            "langfuse_session_id": _langfuse_session_id,
        })
        
        # Langfuse wrapper automatically traces this call with metadata
        response = self._call_completion(messages, name=f"openrouter_{module_name}_image", metadata=metadata)
        result = response.choices[0].message.content

        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")

        return result

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt using OpenRouter API"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]

        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

        # Build metadata with Langfuse session_id, user_id, and game_state
        metadata = _get_langfuse_metadata(game_state)
        metadata.update({
            "backend": "openrouter",
            "module": module_name,
            "has_image": False,
            "langfuse_session_id": _langfuse_session_id,
        })
        
        # Langfuse wrapper automatically traces this call with metadata
        response = self._call_completion(messages, name=f"openrouter_{module_name}_text", metadata=metadata)
        result = response.choices[0].message.content

        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")

        return result

class VLLMBackend(VLMBackend):
    """vLLM or any OpenAI-compatible HTTP endpoint backend with Langfuse tracing"""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        **kwargs,
    ):
        try:
            # Use Langfuse-wrapped OpenAI client if Langfuse is enabled
            if _langfuse_enabled:
                from langfuse.openai import OpenAI
                logger.info("Using Langfuse-wrapped OpenAI client for vLLM tracing")
            else:
                from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name or os.getenv("VLLM_MODEL")
        if not self.model_name:
            raise ValueError("Error: vLLM model name is missing! Set VLLM_MODEL or provide --model-name.")

        resolved_base_url = base_url or os.getenv("VLLM_BASE_URL", "")
        resolved_base_url = resolved_base_url.rstrip("/")
        if not resolved_base_url:
            raise ValueError("Error: vLLM base URL is missing! Set VLLM_BASE_URL environment variable.")

        # Remove trailing /chat/completions if present - OpenAI client adds it
        if resolved_base_url.endswith("/chat/completions"):
            resolved_base_url = resolved_base_url[:-len("/chat/completions")]
        # Also handle /v1 suffix
        if not resolved_base_url.endswith("/v1"):
            resolved_base_url = f"{resolved_base_url}/v1"

        self.api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"

        # Runtime configuration knobs
        timeout = timeout if timeout is not None else os.getenv("VLLM_TIMEOUT")
        self.timeout = float(timeout) if timeout not in (None, "") else 60.0

        temperature = temperature if temperature is not None else os.getenv("VLLM_TEMPERATURE")
        self.temperature = float(temperature) if temperature not in (None, "") else None

        max_tokens = max_tokens if max_tokens is not None else os.getenv("VLLM_MAX_TOKENS")
        self.max_tokens = int(max_tokens) if max_tokens not in (None, "") else None

        top_p = top_p if top_p is not None else os.getenv("VLLM_TOP_P")
        self.top_p = float(top_p) if top_p not in (None, "") else None

        # Initialize OpenAI client with custom base_url for vLLM
        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    @retry_with_exponential_backoff
    def _call_completion(self, messages, **kwargs):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

    def _invoke_model(self, messages: list[dict[str, Any]], prompt_text: str, module_name: str, has_image: bool, game_state: dict = None) -> str:
        start_time = time.time()

        # Build kwargs for the API call
        call_kwargs = {}
        if self.temperature is not None:
            call_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            call_kwargs["top_p"] = self.top_p

        # Build metadata with Langfuse session_id, user_id, and game_state
        metadata = _get_langfuse_metadata(game_state)
        metadata.update({
            "backend": "vllm",
            "module": module_name,
            "has_image": has_image,
            "langfuse_session_id": str(_langfuse_session_id),
        })
        call_kwargs["metadata"] = metadata
        call_kwargs["name"] = f"vllm_{module_name}_{'image' if has_image else 'text'}"

        try:
            # Langfuse wrapper automatically traces this call with metadata
            response = self._call_completion(messages, **call_kwargs)
            result = response.choices[0].message.content
            end_time = time.time()
            duration = end_time - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            log_llm_interaction(
                interaction_type=f"vllm_{module_name}",
                prompt=prompt_text,
                response=result,
                duration=duration,
                metadata={
                    "model": self.model_name,
                    "backend": "vllm",
                    "has_image": has_image,
                    "token_usage": token_usage
                },
                model_info={"model": self.model_name, "backend": "vllm"},
            )

            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"vllm_{module_name}",
                prompt=prompt_text,
                error=str(e),
                metadata={
                    "model": self.model_name,
                    "backend": "vllm",
                    "duration": duration,
                    "has_image": has_image,
                },
            )
            logger.error(f"vLLM error: {e}")
            raise

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt using a vLLM endpoint"""
        if hasattr(img, "convert"):
            image = img
        elif hasattr(img, "shape"):
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": text},
                ],
            }
        ]

        return self._invoke_model(messages, text, module_name, has_image=True, game_state=game_state)

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt using a vLLM endpoint"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        return self._invoke_model(messages, text, module_name, has_image=False, game_state=game_state)

class VertexBackend(VLMBackend):
    """Google Gemini API with Vertex backend"""

    def __init__(self, model_name: str, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")

        self.model_name = model_name

        # Initialize the model
        self.client = genai.Client(
            vertexai=True,
            project=vertex_id,
            location='us-central1',
        )
        self.genai = genai

        # Initialize VertexAI instrumentor if Langfuse is enabled
        self.instrumentor = None
        if _langfuse_enabled:
            try:
                from openinference.instrumentation.vertexai import VertexAIInstrumentor
                self.instrumentor = VertexAIInstrumentor()
                self.instrumentor.instrument()
                logger.info("VertexAI instrumentation initialized for this backend instance")
            except ImportError:
                logger.warning("openinference-instrumentation-vertexai not installed")
            except Exception as e:
                logger.warning(f"Failed to instrument VertexAI: {e}")

        logger.info(f"Vertex backend initialized with model: {model_name}")

    def _prepare_image(self, img: Image.Image | np.ndarray) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content_parts
        )
        return response

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt using Vertex AI"""
        try:
            start_time = time.time()
            image = self._prepare_image(img)

            # Prepare content for Vertex AI
            content_parts = [text, image]

            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] VERTEX AI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Set OpenTelemetry attributes for Langfuse (session_id, user_id, game_state)
            # OpenInference instrumentation will automatically capture this
            if _langfuse_enabled:
                from opentelemetry import trace
                import json
                span = trace.get_current_span()
                if span and span.is_recording():
                    # Set Langfuse-specific attributes
                    if _langfuse_session_id:
                        span.set_attribute("langfuse.session_id", _langfuse_session_id)
                    if _langfuse_user_id:
                        span.set_attribute("langfuse.user_id", _langfuse_user_id)
                    span.set_attribute("langfuse.tags", ["vertex", module_name, "image"])
                    span.set_attribute("module", module_name)
                    span.set_attribute("backend", "vertex")
                    span.set_attribute("has_image", True)
                    
                    # Add game_state if provided
                    if game_state:
                        game_state_metadata = _prepare_game_state_metadata(game_state)
                        if game_state_metadata:
                            # Convert to JSON string for span attribute
                            span.set_attribute("game_state", json.dumps(game_state_metadata))

            # Generate response - OpenInference will automatically trace this
            response = self._call_generate_content(content_parts)

            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)

            result = response.text
            duration = time.time() - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            # Log the interaction
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "vertex", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "vertex"}
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            logger.error(f"Error in Vertex AI image query: {e}")
            # Try text-only fallback for any Vertex error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name, game_state)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt using Vertex AI"""
        try:
            start_time = time.time()
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] VERTEX AI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Set OpenTelemetry attributes for Langfuse (session_id, user_id, game_state)
            # OpenInference instrumentation will automatically capture this
            if _langfuse_enabled:
                from opentelemetry import trace
                import json
                span = trace.get_current_span()
                if span and span.is_recording():
                    # Set Langfuse-specific attributes
                    if _langfuse_session_id:
                        span.set_attribute("langfuse.session_id", _langfuse_session_id)
                    if _langfuse_user_id:
                        span.set_attribute("langfuse.user_id", _langfuse_user_id)
                    span.set_attribute("langfuse.tags", ["vertex", module_name, "text"])
                    span.set_attribute("module", module_name)
                    span.set_attribute("backend", "vertex")
                    span.set_attribute("has_image", False)
                    
                    # Add game_state if provided
                    if game_state:
                        game_state_metadata = _prepare_game_state_metadata(game_state)
                        if game_state_metadata:
                            # Convert to JSON string for span attribute
                            span.set_attribute("game_state", json.dumps(game_state_metadata))

            # Generate response - OpenInference will automatically trace this
            response = self._call_generate_content([text])

            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."

            result = response.text
            duration = time.time() - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            # Log the interaction
            log_llm_interaction(
                interaction_type=f"vertex_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "vertex", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "vertex"}
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            print(f"Error in Gemini text query: {e}")
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class GeminiBackend(VLMBackend):
    """Google Gemini API backend"""

    def __init__(self, model_name: str, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")

        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai

        # Initialize GoogleGenAI instrumentor if Langfuse is enabled
        self.instrumentor = None
        if _langfuse_enabled:
            try:
                from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
                self.instrumentor = GoogleGenAIInstrumentor()
                self.instrumentor.instrument()
                logger.info("GoogleGenAI instrumentation initialized for this backend instance")
            except ImportError:
                logger.warning("openinference-instrumentation-google-genai not installed")
            except Exception as e:
                logger.warning(f"Failed to instrument GoogleGenAI: {e}")

        logger.info(f"Gemini backend initialized with model: {model_name}")

    def _prepare_image(self, img: Image.Image | np.ndarray) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.model.generate_content(content_parts)
        response.resolve()
        return response

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt using Gemini API"""
        start_time = time.time()

        try:
            image = self._prepare_image(img)

            # Prepare content for Gemini
            content_parts = [text, image]

            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Set OpenTelemetry attributes for Langfuse (session_id, user_id, game_state)
            # OpenInference instrumentation will automatically capture this
            if _langfuse_enabled:
                from opentelemetry import trace
                import json
                span = trace.get_current_span()
                if span and span.is_recording():
                    # Set Langfuse-specific attributes
                    if _langfuse_session_id:
                        span.set_attribute("langfuse.session_id", _langfuse_session_id)
                    if _langfuse_user_id:
                        span.set_attribute("langfuse.user_id", _langfuse_user_id)
                    span.set_attribute("langfuse.tags", ["gemini", module_name, "image"])
                    span.set_attribute("module", module_name)
                    span.set_attribute("backend", "gemini")
                    span.set_attribute("has_image", True)
                    
                    # Add game_state if provided
                    if game_state:
                        game_state_metadata = _prepare_game_state_metadata(game_state)
                        if game_state_metadata:
                            # Convert to JSON string for span attribute
                            span.set_attribute("game_state", json.dumps(game_state_metadata))

            # Generate response - OpenInference will automatically trace this
            response = self._call_generate_content(content_parts)

            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name, game_state)

            result = response.text
            end_time = time.time()
            duration = end_time - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Try text-only fallback for any Gemini error
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name, game_state)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt using Gemini API"""
        start_time = time.time()

        try:
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")

            # Set OpenTelemetry attributes for Langfuse (session_id, user_id, game_state)
            # OpenInference instrumentation will automatically capture this
            if _langfuse_enabled:
                from opentelemetry import trace
                import json
                span = trace.get_current_span()
                if span and span.is_recording():
                    # Set Langfuse-specific attributes
                    if _langfuse_session_id:
                        span.set_attribute("langfuse.session_id", _langfuse_session_id)
                    if _langfuse_user_id:
                        span.set_attribute("langfuse.user_id", _langfuse_user_id)
                    span.set_attribute("langfuse.tags", ["gemini", module_name, "text"])
                    span.set_attribute("module", module_name)
                    span.set_attribute("backend", "gemini")
                    span.set_attribute("has_image", False)
                    
                    # Add game_state if provided
                    if game_state:
                        game_state_metadata = _prepare_game_state_metadata(game_state)
                        if game_state_metadata:
                            # Convert to JSON string for span attribute
                            span.set_attribute("game_state", json.dumps(game_state_metadata))

            # Generate response - OpenInference will automatically trace this
            response = self._call_generate_content([text])

            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."

            result = response.text
            end_time = time.time()
            duration = end_time - start_time

            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }

            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )

            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")

            return result

        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."

class VLM:
    """Main VLM class that supports multiple backends"""

    BACKENDS = {
        'openai': OpenAIBackend,
        'openrouter': OpenRouterBackend,
        'vllm': VLLMBackend,
        'gemini': GeminiBackend,
        'vertex': VertexBackend,  # Added Vertex backend
    }

    def __init__(self, model_name: str, backend: str = 'openai', **kwargs):
        """
        Initialize VLM with specified backend

        Args:
            model_name: Name of the model to use
            backend: Backend type ('openai', 'openrouter', 'vllm', 'gemini', 'vertex', or 'auto')
            **kwargs: Additional arguments passed to backend
        """
        self.model_name = model_name
        self.backend_type = backend.lower()

        # Auto-detect backend based on model name if not explicitly specified
        if backend == 'auto':
            self.backend_type = self._auto_detect_backend(model_name)

        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")

        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]

        self.backend = backend_class(model_name, **kwargs)

        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")

    def _auto_detect_backend(self, model_name: str) -> str:
        """Auto-detect backend based on model name"""
        model_lower = model_name.lower()

        if any(x in model_lower for x in ['gpt', 'o4-mini', 'o3', 'claude']):
            return 'openai'
        elif any(x in model_lower for x in ['gemini', 'palm']):
            return 'gemini'
        elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'phi']):
            return 'local'
        else:
            # Default to OpenAI for unknown models
            return 'openai'

    def get_query(self, img: Image.Image | np.ndarray, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process an image and text prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_query(img, text, module_name, game_state=game_state)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": True}
            )
            raise

    def get_text_query(self, text: str, module_name: str = "Unknown", game_state: dict = None) -> str:
        """Process a text-only prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_text_query(text, module_name, game_state=game_state)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": False}
            )
            raise
