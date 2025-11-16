"""
Image embedding utilities using Azure Computer Vision API.

This module provides functions to generate image embeddings for use in
vector similarity search and memory retrieval.
"""
import os
import io
import numpy as np
import requests
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """
    Convert numpy array frame to PNG bytes for API submission.

    Args:
        frame: Numpy array representing the image

    Returns:
        PNG encoded bytes
    """
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def get_image_embedding_from_bytes(
    image_data: bytes,
    image_description: str = "image"
) -> Optional[np.ndarray]:
    """
    Get image embedding vector using Azure Computer Vision API's vectorizeImage endpoint.

    Args:
        image_data: Binary image data (PNG/JPEG bytes)
        image_description: Description for logging purposes

    Returns:
        Embedding vector as numpy array, or None if request fails
    """
    endpoint = os.environ.get("VISION_ENDPOINT")
    key = os.environ.get("VISION_KEY")

    if not endpoint or not key:
        logger.warning("VISION_ENDPOINT or VISION_KEY not set, cannot generate embeddings")
        return None

    logger.debug(f"Getting image embedding for: {image_description}")

    # Call API with versioning (required based on testing)
    url = f"{endpoint}/computervision/retrieval:vectorizeImage?api-version=2024-02-01&model-version=2023-04-15"

    headers = {
        "Content-Type": "application/octet-stream",
        "Ocp-Apim-Subscription-Key": key
    }

    try:
        response = requests.post(url, headers=headers, data=image_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # The API returns a vector in the response
            if 'vector' in result:
                embedding = np.array(result['vector'])
                logger.debug(f"Successfully retrieved embedding of dimension {len(embedding)}")
                return embedding
            else:
                logger.warning(f"Unexpected response format: {result.keys()}")
                return None
        else:
            logger.error(f"Error calling vectorizeImage API: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception during embedding generation: {e}")
        return None


def get_image_embedding_from_frame(
    frame: np.ndarray,
    image_description: str = "frame"
) -> Optional[np.ndarray]:
    """
    Get image embedding from a numpy array frame.

    This is the primary function to use for generating embeddings from game frames.

    Args:
        frame: Numpy array representing the image (shape: H x W x 3)
        image_description: Description for logging purposes

    Returns:
        Embedding vector as numpy array (1024-dimensional), or None if request fails

    Example:
        >>> frame = np.array(game_state['frame'])
        >>> embedding = get_image_embedding_from_frame(frame, "current game state")
        >>> if embedding is not None:
        >>>     print(f"Generated {len(embedding)}-dimensional embedding")
    """
    # Convert frame to PNG bytes
    image_data = frame_to_png_bytes(frame)
    return get_image_embedding_from_bytes(image_data, image_description)


def get_image_embedding_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Get image embedding vector from a file path.

    Args:
        image_path: Path to the image file

    Returns:
        Embedding vector as numpy array, or None if request fails
    """
    # Read image as binary data
    with open(image_path, "rb") as f:
        image_data = f.read()

    return get_image_embedding_from_bytes(image_data, image_path)


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1 (typically 0 to 1 for embeddings)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
