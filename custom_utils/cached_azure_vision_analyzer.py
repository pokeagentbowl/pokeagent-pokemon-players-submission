"""Cached wrapper for Azure Computer Vision API with per-feature caching support."""
import hashlib
import json
import logging
from typing import List, Optional, Dict
import numpy as np

from azure.ai.vision.imageanalysis.models import VisualFeatures

from custom_utils.azure_vision_analyzer import AzureVisionAnalyzer
from custom_utils.cache_utils import get_object_detector_cache

logger = logging.getLogger(__name__)


class CachedAzureVisionAnalyzer(AzureVisionAnalyzer):
    """
    Cached wrapper for Azure Computer Vision API with per-feature caching.

    Supports independent caching of visual features (OBJECTS, DENSE_CAPTIONS, PEOPLE, READ)
    so that only uncached features are computed on subsequent requests.

    Uses the same cache namespace as CachedObjectDetector (vision_{image_hash}_{feature})
    to ensure cache sharing across different analyzer classes.
    """

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None,
                 save_debug_frames: bool = False, debug_output_dir: str = "debug_frames",
                 cache_dir: Optional[str] = None, cache=None, prefer_redis: bool = True):
        """
        Initialize cached vision analyzer with unified Store interface.

        Attempts to use RedisStore first (if credentials available), falls back to LocalFileStore.
        Both cache types use the same Store interface (mget/mset), eliminating conditional logic.

        Args:
            endpoint: Azure Computer Vision endpoint
            key: Azure Computer Vision API key
            save_debug_frames: Whether to save annotated frames
            debug_output_dir: Directory to save debug frames
            cache_dir: Optional custom cache directory (defaults to .custom_cache)
            cache: Optional pre-initialized cache instance (RedisStore or LocalFileStore)
            prefer_redis: If True, attempts Redis cache first before falling back to LocalFileStore
        """
        super().__init__(endpoint, key, save_debug_frames, debug_output_dir)

        # Setup cache using centralized cache utility
        if cache is not None:
            self.cache = cache
        elif cache_dir:
            self.cache = get_object_detector_cache(cache_dir=cache_dir, prefer_redis=prefer_redis)
        else:
            self.cache = get_object_detector_cache(prefer_redis=prefer_redis)

        # Both RedisStore and LocalFileStore use the same Store interface
        # No need to detect cache type or use conditionals!
        logger.info(f"Initialized CachedAzureVisionAnalyzer with {type(self.cache).__name__}")

    def _generate_image_hash(self, frame: np.ndarray) -> str:
        """
        Generate unique hash for an image frame.

        Args:
            frame: Numpy array representing the image

        Returns:
            SHA256 hash of the image data
        """
        # Convert frame to bytes for hashing
        image_bytes = self._frame_to_png_bytes(frame)
        return hashlib.sha256(image_bytes).hexdigest()

    def _generate_cache_key(self, image_hash: str, feature: str) -> str:
        """
        Generate cache key for a specific feature on an image.

        IMPORTANT: Uses the same namespace as CachedObjectDetector to enable cache sharing.

        Args:
            image_hash: Hash of the image
            feature: Visual feature name (e.g., 'OBJECTS', 'DENSE_CAPTIONS', 'PEOPLE', 'READ')

        Returns:
            Cache key string in format: vision_{image_hash}_{feature}
        """
        return f"vision_{image_hash}_{feature}"

    def _cache_lookup(self, cache_key: str) -> Optional[str]:
        """
        Look up a value in the cache using unified Store interface.

        Both RedisStore and LocalFileStore use the same mget API.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached value as string, or None if not found
        """
        # Use Store API: mget returns list of bytes or None
        cached_values = self.cache.mget([cache_key])
        cached_value = cached_values[0] if cached_values and cached_values[0] else None
        if cached_value:
            return cached_value.decode('utf-8')
        return None

    def _cache_update(self, cache_key: str, value: str):
        """
        Update a value in the cache using unified Store interface.

        Both RedisStore and LocalFileStore use the same mset API.

        Args:
            cache_key: Cache key to update
            value: Value to cache (as string)
        """
        # Use Store API: mset expects list of (key, value) tuples with bytes
        self.cache.mset([(cache_key, value.encode('utf-8'))])

    def _get_analysis_result(self, scaled_frame: np.ndarray,
                            visual_features: Optional[List[VisualFeatures]] = None) -> Dict:
        """
        Get analysis result with per-feature caching.

        Overrides parent method to add caching logic. Checks cache for each feature
        and only calls API for missing features.

        Args:
            scaled_frame: Scaled frame ready for API submission
            visual_features: Optional list of visual features to request

        Returns:
            Complete analysis result dict with all requested features (cached + fresh)
        """
        # Default to all four features (objects + OCR) to match parent class behavior
        if visual_features is None:
            visual_features = [
                VisualFeatures.OBJECTS,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.PEOPLE,
                VisualFeatures.READ
            ]

        # Generate image hash for cache key
        image_hash = self._generate_image_hash(scaled_frame)

        # Check cache for each requested feature
        complete_result_dict = {}
        missing_features = []

        for feature in visual_features:
            feature_name = feature.name if hasattr(feature, 'name') else str(feature)
            cache_key = self._generate_cache_key(image_hash, feature_name)
            cached_value = self._cache_lookup(cache_key)

            if cached_value:
                logger.debug(f"Cache HIT for feature: {feature_name}")
                # Merge cached feature data into complete dict
                complete_result_dict.update(json.loads(cached_value))
            else:
                logger.debug(f"Cache MISS for feature: {feature_name}")
                missing_features.append(feature)

        # Call parent method for missing features (handles API call)
        if missing_features:
            logger.debug(f"Computing {len(missing_features)} missing features via API")

            # Call parent's method to get fresh data from Azure API
            fresh_result_dict = super()._get_analysis_result(scaled_frame, missing_features)

            # Cache each computed feature separately
            for feature in missing_features:
                feature_name = feature.name if hasattr(feature, 'name') else str(feature)
                cache_key = self._generate_cache_key(image_hash, feature_name)

                # Extract feature-specific result for caching
                feature_data = self._extract_feature_from_dict(fresh_result_dict, feature)
                serialized_data = json.dumps(feature_data)
                self._cache_update(cache_key, serialized_data)

                # Merge into complete result dict
                complete_result_dict.update(feature_data)
        else:
            logger.debug("All features retrieved from cache, no API call needed")

        return complete_result_dict

    def _extract_feature_from_dict(self, result_dict: Dict, feature: VisualFeatures) -> Dict:
        """
        Extract feature-specific data from result dict for caching.

        Args:
            result_dict: Azure analysis result dict (from result.as_dict())
            feature: Visual feature to extract

        Returns:
            Dictionary containing only the specified feature data
        """
        feature_name = feature.name if hasattr(feature, 'name') else str(feature)

        # Map feature names to their corresponding result keys in Azure's dict format
        feature_key_map = {
            "OBJECTS": "objectsResult",
            "DENSE_CAPTIONS": "denseCaptionsResult",
            "PEOPLE": "peopleResult",
            "READ": "readResult"
        }

        result_key = feature_key_map.get(feature_name)
        if result_key and result_key in result_dict:
            return {result_key: result_dict[result_key]}

        # Return empty result structure if feature not found
        if result_key == "readResult":
            return {result_key: {"blocks": []}}
        else:
            return {result_key: {"values": []}} if result_key else {}
