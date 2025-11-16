import os
import logging

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def create_llm(
    llm_type: str, 
    model_name: str = None, 
    temperature: float = 0,
    request_timeout: int = 300,
    verbose: bool = True,
    callbacks=None,
    max_retries: int = 5,
    seed: int = 42,
    **kwargs
) -> BaseChatModel:
    """Factory function to create the appropriate LLM
    Modern version of construct_chat_model
    
    Supported llm_types:
    - 'ollama': Local Ollama models
    - 'groq': Groq API models  
    - 'azure_openai': Azure OpenAI models
    - 'azure_openai_v1': Azure OpenAI with v1 endpoint
    - 'openai': OpenAI API models
    - 'github_models': GitHub Models (free, rate-limited testing)
    
    For GitHub Models:
    - Uses AZURE_INFERENCE_CREDENTIAL and AZURE_INFERENCE_ENDPOINT environment variables
    - Recommended model: 'gpt-4o-mini' for 
        - low tier rate limits 
        - structured output + tool calling support
    - Suitable for testing only due to rate/token limits
    """
    logger.info(f"Using model: {model_name}")

    model_params = {
        'model': model_name,
        'temperature': temperature,
        'request_timeout': request_timeout,
        'verbose': verbose,
        'callbacks': callbacks,
        'seed': seed,
    }
    if max_retries:
        model_params['max_retries'] = max_retries
    
    logger.info(f"Using {llm_type}")
    if llm_type == 'ollama':    
        from langchain_ollama.chat_models import ChatOllama
        chat_model_class = ChatOllama
    elif llm_type == 'groq':
        from langchain_groq import ChatGroq
        # set GROQ_API_KEY env var
        chat_model_class = ChatGroq
    elif llm_type == 'azure_openai':
        from langchain_openai import AzureChatOpenAI
        chat_model_class = AzureChatOpenAI
        # assumes deployment name is same as model name, might be different!
    elif llm_type == 'azure_openai_v1':
        from langchain_openai import ChatOpenAI
        chat_model_class = ChatOpenAI
        base_url = os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('OPENAI_BASE_URL')
        base_url = base_url.rstrip('/')
        # Ensure URL ends with /openai/v1/
        if not base_url.endswith('/openai/v1'):
            base_url += '/openai/v1/'
        else:
            base_url += '/'
        model_params['base_url'] = base_url
        model_params['api_key'] = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        # remember to set base url as the Azure OpenAI endpoint with /openai/v1 appended
        # and OPENAI_API_KEY as the Azure OpenAI API key
    elif llm_type == 'openai':
        from langchain_openai import ChatOpenAI
        chat_model_class = ChatOpenAI

    elif llm_type == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        chat_model_class = ChatGoogleGenerativeAI
        # Configure for Vertex AI
        model_params['vertexai'] = True
        model_params['project'] = os.getenv('GOOGLE_CLOUD_PROJECT') or 'pokeagent-007'
        model_params['location'] = os.getenv('GOOGLE_CLOUD_LOCATION') or 'us-central1'
        # Credentials will use application default if not specified

    elif llm_type == 'vllm':
        from langchain_openai import ChatOpenAI
        chat_model_class = ChatOpenAI

        resolved_model_name = model_name or os.getenv('AGENT_MODEL_NAME')
        if not resolved_model_name:
            raise ValueError("Error: vLLM model name is missing! Set AGENT_MODEL_NAME or provide model_name.")

        base_url = kwargs.pop('base_url', None) or os.getenv('VLLM_BASE_URL', '')
        if not base_url:
            raise ValueError("Error: vLLM base URL is missing! Set VLLM_BASE_URL environment variable or pass base_url.")

        base_url = base_url.rstrip('/')
        if base_url.endswith('/chat/completions'):
            base_url = base_url[:-len('/chat/completions')]
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"  # Reason: LangChain client expects the OpenAI-compatible /v1 route.

        api_key = kwargs.pop('api_key', None) or os.getenv('VLLM_API_KEY') or "EMPTY"

        timeout_override = (
            kwargs.pop('timeout', None)
            or kwargs.pop('request_timeout', None)
            or os.getenv('VLLM_TIMEOUT')
        )
        if timeout_override not in (None, ''):
            timeout_value = float(timeout_override)
            model_params['timeout'] = timeout_value
            model_params['request_timeout'] = timeout_value

        if temperature is None:
            temp_override = os.getenv('VLLM_TEMPERATURE')
            if temp_override not in (None, ''):
                model_params['temperature'] = float(temp_override)

        max_tokens_override = kwargs.pop('max_tokens', None) or os.getenv('VLLM_MAX_TOKENS')
        if max_tokens_override not in (None, ''):
            model_params['max_tokens'] = int(max_tokens_override)

        top_p_override = kwargs.pop('top_p', None) or os.getenv('VLLM_TOP_P')
        if top_p_override not in (None, ''):
            model_params['top_p'] = float(top_p_override)

        model_params.update({
            'model': resolved_model_name,
            'base_url': base_url,
            'api_key': api_key,
        })

    elif llm_type == 'github_models':
        # Used for testing only
        from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
        # Uses AZURE_INFERENCE_CREDENTIAL and AZURE_INFERENCE_ENDPOINT by default
        # NOTE: recent changes 0.1.6 now use AZURE_AI_CREDENTIAL and AZURE_AI_ENDPOINT
        chat_model_class = AzureAIChatCompletionsModel
        model_params.update({
            'model': model_name or 'gpt-4o-mini',  # Default to gpt-4o-mini for rate limits
            'api_version': '2024-08-01-preview',  # Required for structured output support
            'endpoint': os.getenv('AZURE_INFERENCE_ENDPOINT') or os.getenv('AZURE_AI_ENDPOINT'),
            'credential': os.getenv('AZURE_INFERENCE_CREDENTIAL') or os.getenv('AZURE_AI_CREDENTIAL'),
            'max_tokens': 3500  # 4k limit for free tier
        })
        # Remove 'model' from model_params to avoid duplication (it's set above)
        if 'model' in model_params:
            del model_params['model']
        # Set the model parameter directly for AzureAIChatCompletionsModel
        model_params['model'] = model_name or 'gpt-4o-mini'
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    constructed_chat_model = chat_model_class(**model_params)
    return constructed_chat_model
