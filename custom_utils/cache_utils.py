"""Centralized cache utilities for LangChain caching (Redis and SQLite)."""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Union

from langchain_community.cache import SQLiteCache, RedisCache
from langchain_core.globals import set_llm_cache

logger = logging.getLogger(__name__)

# Default cache directory for all caches
DEFAULT_CACHE_DIR = ".custom_cache"
DEFAULT_CACHE_DB = "langchain_cache.db"

# Cache instances by path
_cache_instances: Dict[str, SQLiteCache] = {}
_redis_cache_instance: Optional[RedisCache] = None


def get_langchain_sqlite_cache(
    cache_dir: str = DEFAULT_CACHE_DIR,
    db_name: str = DEFAULT_CACHE_DB
) -> SQLiteCache:
    """
    Get or create a shared SQLiteCache instance.
    
    This provides a unified cache for both LLM calls and custom caching needs
    (e.g., Azure Computer Vision API). Multiple cache paths are supported,
    but by default all caches share the same database for consistency.
    
    Args:
        cache_dir: Directory to store cache database (default: .custom_cache)
        db_name: Name of the cache database file (default: langchain_cache.db)
        
    Returns:
        SQLiteCache instance
    """
    cache_path = os.path.join(cache_dir, db_name)
    
    # Return existing instance if already created
    if cache_path in _cache_instances:
        return _cache_instances[cache_path]
    
    # Create new instance
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = SQLiteCache(database_path=cache_path)
    _cache_instances[cache_path] = cache
    logger.info(f"LangChain SQLite cache initialized at {cache_path}")
    
    return cache


# For backwards compatibility
def get_langchain_cache(
    cache_dir: str = DEFAULT_CACHE_DIR,
    db_name: str = DEFAULT_CACHE_DB
) -> SQLiteCache:
    """
    Deprecated: Use get_langchain_sqlite_cache() instead.
    Maintained for backwards compatibility.
    """
    return get_langchain_sqlite_cache(cache_dir, db_name)


def get_langchain_redis_cache() -> Optional[RedisCache]:
    """
    Get or create a Redis cache instance for LangChain.
    
    Connects to Azure Redis using Entra ID authentication.
    Requires environment variables:
    - ENTRAID_CLIENT_ID: Service principal client ID
    - ENTRAID_CLIENT_SECRET: Service principal secret
    - ENTRAID_TENANT_ID: Azure tenant ID
    - REDIS_HOST: Redis hostname
    
    Returns:
        RedisCache instance if connection successful, None otherwise
    """
    global _redis_cache_instance
    
    # Return existing instance if already created
    if _redis_cache_instance is not None:
        return _redis_cache_instance
    
    # Check for required environment variables
    client_id = os.getenv("ENTRAID_CLIENT_ID")
    client_secret = os.getenv("ENTRAID_CLIENT_SECRET")
    tenant_id = os.getenv("ENTRAID_TENANT_ID")
    redis_host = os.getenv("REDIS_HOST")
    
    if not all([client_id, client_secret, tenant_id, redis_host]):
        logger.info("Redis cache environment variables not set, skipping Redis cache")
        return None
    
    try:
        from redis import Redis
        from redis_entraid.cred_provider import create_from_service_principal
        
        # Create Entra ID credential provider
        credential_provider = create_from_service_principal(
            client_id,
            client_secret,
            tenant_id
        )
        
        # Create Redis client
        redis_client = Redis(
            host=redis_host,
            port=6380,  # Azure Cache for Redis classic uses port 6380
            ssl=True,
            credential_provider=credential_provider,
            decode_responses=True,
            socket_timeout=30,
            socket_connect_timeout=30
        )
        
        # Test connection
        redis_client.ping()
        
        # Create and cache RedisCache instance
        _redis_cache_instance = RedisCache(redis_client)
        logger.info(f"LangChain Redis cache initialized at {redis_host}")
        
        return _redis_cache_instance
        
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache: {e}. Will use SQLite cache as fallback.")
        return None


def init_langchain_llm_cache(
    cache_dir: str = DEFAULT_CACHE_DIR,
    db_name: str = DEFAULT_CACHE_DB,
    prefer_redis: bool = True
) -> None:
    """
    Initialize LangChain's global LLM cache using set_llm_cache().
    
    Attempts to use Redis cache first if environment variables are set.
    Falls back to SQLite cache if Redis is unavailable.
    
    This should be called once during agent initialization to enable automatic
    caching for all LLM calls made through LangChain chat models.
    
    Args:
        cache_dir: Directory to store SQLite cache database (default: .custom_cache)
        db_name: Name of the SQLite cache database file (default: langchain_cache.db)
        prefer_redis: If True, attempts Redis cache first before falling back to SQLite
    """
    cache = None
    
    # Try Redis cache first if preferred
    if prefer_redis:
        cache = get_langchain_redis_cache()
        if cache is not None:
            set_llm_cache(cache)
            logger.info("LangChain global LLM cache configured with Redis")
            return
    
    # Fallback to SQLite cache
    cache = get_langchain_sqlite_cache(cache_dir, db_name)
    set_llm_cache(cache)
    logger.info("LangChain global LLM cache configured with SQLite")


def create_cached_embeddings(
    cache_dir: str = DEFAULT_CACHE_DIR,
    prefer_redis: bool = True
):
    """
    Create cache-backed embeddings with Redis or LocalFileStore fallback.
    
    This function attempts to use Redis cache first if environment variables are set,
    falling back to LocalFileStore if Redis is unavailable. It follows the same pattern
    as init_langchain_llm_cache() for consistency.
    
    The embeddings use ImageFrameEmbeddings from custom_utils.custom_embeddings for
    Azure Computer Vision API image embeddings.
    
    Args:
        cache_dir: Directory to store LocalFileStore cache (default: .custom_cache)
        prefer_redis: If True, attempts Redis cache first before falling back to LocalFileStore
        
    Returns:
        CacheBackedEmbeddings instance configured with either Redis or LocalFileStore
        
    Example:
        >>> embeddings = create_cached_embeddings()
        >>> # Use with vector store
        >>> from langchain_chroma import Chroma
        >>> vectorstore = Chroma(
        ...     embedding_function=embeddings,
        ...     persist_directory="./chroma_db"
        ... )
    """
    from custom_utils.custom_embeddings import ImageFrameEmbeddings
    
    # Import with fallback for different langchain versions
    try:
        from langchain_classic.embeddings import CacheBackedEmbeddings
        from langchain_classic.storage import LocalFileStore
    except ImportError:
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain.storage import LocalFileStore
    
    # Create underlying embeddings
    underlying_embeddings = ImageFrameEmbeddings()
    
    # Try Redis cache first if preferred
    if prefer_redis:
        redis_embeddings = _try_redis_cached_embeddings(underlying_embeddings)
        if redis_embeddings is not None:
            return redis_embeddings
    
    # Fallback to LocalFileStore
    logger.info("Using LocalFileStore for embeddings cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    store = LocalFileStore(cache_dir)
    
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace="azure_vision_image_embeddings",
        query_embedding_cache=True
    )
    
    logger.info(f"Created cache-backed embeddings with LocalFileStore at {cache_dir}")
    return cached_embeddings


def _try_redis_cached_embeddings(underlying_embeddings):
    """
    Internal helper to try creating Redis-backed embeddings.
    
    Returns CacheBackedEmbeddings with Redis if successful, None otherwise.
    """
    # Import with fallback for different langchain versions
    try:
        from langchain_classic.embeddings import CacheBackedEmbeddings
    except ImportError:
        from langchain.embeddings import CacheBackedEmbeddings
    
    from langchain_community.storage import RedisStore
    
    # Check for required environment variables
    client_id = os.getenv("ENTRAID_CLIENT_ID")
    client_secret = os.getenv("ENTRAID_CLIENT_SECRET")
    tenant_id = os.getenv("ENTRAID_TENANT_ID")
    redis_host = os.getenv("REDIS_HOST")
    
    if not all([client_id, client_secret, tenant_id, redis_host]):
        logger.info("Redis environment variables not set, using LocalFileStore for embeddings")
        return None
    
    try:
        from redis import Redis
        from redis_entraid.cred_provider import create_from_service_principal
        
        # Create Entra ID credential provider
        credential_provider = create_from_service_principal(
            client_id,
            client_secret,
            tenant_id
        )
        
        # Create Redis client
        redis_client = Redis(
            host=redis_host,
            port=6380,  # Azure Cache for Redis classic uses port 6380
            ssl=True,
            credential_provider=credential_provider,
            decode_responses=False,  # Important: must be False for binary embedding data
            socket_timeout=30,
            socket_connect_timeout=30
        )
        
        # Test connection
        redis_client.ping()
        
        # Create RedisStore
        store = RedisStore(
            client=redis_client,
            namespace="azure_vision_embeddings"
        )
        
        # Create cache-backed embeddings
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            store,
            namespace="azure_vision_image_embeddings",
            query_embedding_cache=True
        )
        
        logger.info(f"Created cache-backed embeddings with Redis at {redis_host}")
        return cached_embeddings
        
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache for embeddings: {e}. Will use LocalFileStore.")
        return None


def get_object_detector_cache(
    cache_dir: str = DEFAULT_CACHE_DIR,
    db_name: str = "vision_cache.db",
    prefer_redis: bool = True
):
    """
    Get cache for object detector with Redis or LocalFileStore fallback.

    This function attempts to use RedisStore first if environment variables are set,
    falling back to LocalFileStore if Redis is unavailable. This provides interface
    unification since both RedisStore and LocalFileStore use the same Store API
    (mget/mset), eliminating the need for conditionals.

    The returned cache uses the Store interface:
    - RedisStore (preferred): Uses mget/mset for key-value storage
    - LocalFileStore (fallback): Uses mget/mset for key-value storage

    This is consistent with the cache-backed embeddings pattern.

    Args:
        cache_dir: Directory to store LocalFileStore cache (default: .custom_cache)
        db_name: Ignored (kept for backwards compatibility)
        prefer_redis: If True, attempts Redis cache first before falling back to LocalFileStore

    Returns:
        Either RedisStore instance or LocalFileStore instance (both use Store interface)

    Example:
        >>> from custom_utils.cached_object_detector import CachedObjectDetector
        >>> cache = get_object_detector_cache()
        >>> detector = CachedObjectDetector(cache=cache)
    """
    # Import with fallback for different langchain versions
    try:
        from langchain_classic.storage import LocalFileStore
    except ImportError:
        from langchain.storage import LocalFileStore

    # Try Redis cache first if preferred
    if prefer_redis:
        redis_store = _try_redis_store_for_vision()
        if redis_store is not None:
            return redis_store

    # Fallback to LocalFileStore (unified Store interface)
    logger.info("Using LocalFileStore for object detector")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return LocalFileStore(cache_dir)


def _try_redis_store_for_vision():
    """
    Internal helper to try creating RedisStore for vision caching.
    
    Returns RedisStore if successful, None otherwise.
    """
    from langchain_community.storage import RedisStore
    
    # Check for required environment variables
    client_id = os.getenv("ENTRAID_CLIENT_ID")
    client_secret = os.getenv("ENTRAID_CLIENT_SECRET")
    tenant_id = os.getenv("ENTRAID_TENANT_ID")
    redis_host = os.getenv("REDIS_HOST")
    
    if not all([client_id, client_secret, tenant_id, redis_host]):
        logger.info("Redis environment variables not set, using SQLite cache for object detector")
        return None
    
    try:
        from redis import Redis
        from redis_entraid.cred_provider import create_from_service_principal
        
        # Create Entra ID credential provider
        credential_provider = create_from_service_principal(
            client_id,
            client_secret,
            tenant_id
        )
        
        # Create Redis client
        redis_client = Redis(
            host=redis_host,
            port=6380,  # Azure Cache for Redis classic uses port 6380
            ssl=True,
            credential_provider=credential_provider,
            decode_responses=False,  # Important: must be False for binary data
            socket_timeout=30,
            socket_connect_timeout=30
        )
        
        # Test connection
        redis_client.ping()
        
        # Create RedisStore
        store = RedisStore(
            client=redis_client,
            namespace="azure_vision_object_detection"
        )
        
        logger.info(f"Created RedisStore for object detector at {redis_host}")
        return store
        
    except Exception as e:
        logger.warning(f"Failed to initialize Redis cache for object detector: {e}. Will use SQLite.")
        return None

