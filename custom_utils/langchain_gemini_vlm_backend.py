import logging
import os
import base64

import numpy as np

from io import BytesIO
from PIL import Image
from typing import Union

from utils.vlm import VLMBackend, retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class GeminiBackend(VLMBackend):
    """Google Gemini API backend using langchain"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
        except ImportError:
            raise ImportError("LangChain Google GenAI package not found. Install with: pip install langchain-google-genai")
        
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Initialize the langchain ChatGoogleGenerativeAI model
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            **kwargs
        )
        self.HumanMessage = HumanMessage
        
        logger.info(f"Gemini backend initialized with langchain model: {model_name}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, messages):
        """Calls the langchain chat model with exponential backoff."""
        return self.chat_model.invoke(messages)
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API via langchain"""
        try:
            image = self._prepare_image(img)
            
            # Convert image to base64 for langchain
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create HumanMessage with multimodal content
            message = self.HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ])
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([message])
            result = response.content
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini langchain image query: {e}")
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API via langchain"""
        try:
            # Create HumanMessage with text-only content
            message = self.HumanMessage(content=text)
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([message])
            result = response.content
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini langchain text query: {e}")
            raise
