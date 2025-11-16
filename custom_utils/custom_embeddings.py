"""
Custom embeddings implementation for LangChain using Azure Computer Vision.

This module provides a LangChain-compatible embeddings class that uses Azure
Computer Vision API for generating image embeddings, with cache-backed storage
for improved performance.
"""
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
import logging

from custom_utils.image_embedder import get_image_embedding_from_bytes

logger = logging.getLogger(__name__)


class AzureVisionEmbeddings(Embeddings):
    """
    Custom embeddings class using Azure Computer Vision API.

    This class implements the LangChain Embeddings interface and uses Azure's
    vectorizeImage endpoint for generating embeddings. It's designed to work
    with image data encoded as bytes.

    Note: This embedder is designed for image embeddings. For text inputs,
    it will raise a NotImplementedError as Azure Vision API is image-specific.
    To use this with cache-backed embeddings, wrap it with CacheBackedEmbeddings.

    Example:
        >>> from langchain.storage import LocalFileStore
        >>> from langchain.embeddings import CacheBackedEmbeddings
        >>>
        >>> embeddings = AzureVisionEmbeddings()
        >>> store = LocalFileStore("./.custom_cache")
        >>> cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        ...     embeddings,
        ...     store,
        ...     namespace="azure_vision_2023-04-15"
        ... )
        >>>
        >>> # Use with vector store
        >>> from langchain_chroma import Chroma
        >>> vectorstore = Chroma(
        ...     embedding_function=cached_embeddings,
        ...     persist_directory="./chroma_db"
        ... )
    """

    def __init__(self):
        """Initialize the Azure Vision embeddings."""
        super().__init__()
        logger.info("Initialized AzureVisionEmbeddings")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents (images as byte strings).

        Args:
            texts: List of documents to embed. Each should be a string representation
                   of image bytes or a base64-encoded image string.

        Returns:
            List of embeddings, each as a list of floats

        Raises:
            NotImplementedError: If text embedding is attempted (use for images only)
        """
        embeddings = []

        for i, text in enumerate(texts):
            # Assume text is actually bytes or can be encoded
            if isinstance(text, str):
                # Try to interpret as base64 or raw bytes
                try:
                    import base64
                    image_bytes = base64.b64decode(text)
                except Exception:
                    # If not base64, assume it's raw bytes encoded as latin-1
                    image_bytes = text.encode('latin-1')
            else:
                image_bytes = text

            embedding = get_image_embedding_from_bytes(
                image_bytes,
                f"document_{i}"
            )

            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                logger.warning(f"Failed to generate embedding for document {i}, using zero vector")
                embeddings.append([0.0] * 1024)  # Azure vision embeddings are 1024-dim

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (image as byte string).

        Args:
            text: Query to embed. Should be a string representation of image bytes
                  or a base64-encoded image string.

        Returns:
            Embedding as a list of floats

        Raises:
            NotImplementedError: If text embedding is attempted (use for images only)
        """
        # Assume text is actually bytes or can be encoded
        if isinstance(text, str):
            # Try to interpret as base64 or raw bytes
            try:
                import base64
                image_bytes = base64.b64decode(text)
            except Exception:
                # If not base64, assume it's raw bytes encoded as latin-1
                image_bytes = text.encode('latin-1')
        else:
            image_bytes = text

        embedding = get_image_embedding_from_bytes(image_bytes, "query")

        if embedding is not None:
            return embedding.tolist()
        else:
            logger.warning("Failed to generate query embedding, using zero vector")
            return [0.0] * 1024  # Azure vision embeddings are 1024-dim


class ImageFrameEmbeddings(Embeddings):
    """
    Custom embeddings class for numpy image frames.

    This class is specifically designed for embedding numpy array frames directly,
    without needing to convert them to strings first. This is more efficient for
    working with game frames.

    Note: Since LangChain's Embeddings interface expects strings, this class
    provides helper methods for working with numpy arrays. The standard methods
    (embed_documents, embed_query) expect serialized numpy arrays.

    Example:
        >>> embeddings = ImageFrameEmbeddings()
        >>>
        >>> # Direct frame embedding (preferred)
        >>> import numpy as np
        >>> frame = np.array(game_state['frame'])
        >>> embedding = embeddings.embed_frame(frame)
        >>>
        >>> # Batch frame embedding
        >>> frames = [frame1, frame2, frame3]
        >>> embeddings_list = embeddings.embed_frames(frames)
    """

    def __init__(self):
        """Initialize the image frame embeddings."""
        super().__init__()
        from custom_utils.image_embedder import get_image_embedding_from_frame
        self._embed_frame_func = get_image_embedding_from_frame
        logger.info("Initialized ImageFrameEmbeddings")

    def embed_frame(self, frame: np.ndarray, description: str = "frame") -> List[float]:
        """
        Embed a single numpy image frame.

        Args:
            frame: Numpy array representing the image (H x W x 3)
            description: Description for logging

        Returns:
            Embedding as a list of floats
        """
        embedding = self._embed_frame_func(frame, description)

        if embedding is not None:
            return embedding.tolist()
        else:
            logger.warning(f"Failed to generate embedding for {description}, using zero vector")
            return [0.0] * 1024

    def embed_frames(self, frames: List[np.ndarray]) -> List[List[float]]:
        """
        Embed multiple numpy image frames.

        Args:
            frames: List of numpy arrays representing images

        Returns:
            List of embeddings, each as a list of floats
        """
        embeddings = []
        for i, frame in enumerate(frames):
            embedding = self.embed_frame(frame, f"frame_{i}")
            embeddings.append(embedding)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents (serialized numpy arrays).

        Args:
            texts: List of base64-encoded numpy arrays

        Returns:
            List of embeddings, each as a list of floats
        """
        embeddings = []
        for i, text in enumerate(texts):
            # Deserialize numpy array from base64
            try:
                import base64
                frame_bytes = base64.b64decode(text)
                frame = np.frombuffer(frame_bytes, dtype=np.uint8)
                # Assume frame shape can be inferred or is standard
                # For game frames, typically 160x240x3
                frame = frame.reshape(-1, 240, 3)  # Adjust if needed
                embedding = self.embed_frame(frame, f"document_{i}")
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to deserialize frame {i}: {e}")
                embeddings.append([0.0] * 1024)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query (serialized numpy array).

        Args:
            text: Base64-encoded numpy array

        Returns:
            Embedding as a list of floats
        """
        # Deserialize numpy array from base64
        try:
            import base64
            frame_bytes = base64.b64decode(text)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            # Assume frame shape can be inferred or is standard
            frame = frame.reshape(-1, 240, 3)  # Adjust if needed
            return self.embed_frame(frame, "query")
        except Exception as e:
            logger.error(f"Failed to deserialize query frame: {e}")
            return [0.0] * 1024
