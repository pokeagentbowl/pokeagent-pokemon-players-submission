"""
Chroma-compatible embedding function for Azure Computer Vision.

This module provides an embedding function that integrates Azure Computer Vision
image embeddings with Chroma vector database.
"""
import numpy as np
import logging
from typing import List, Any
from langchain_core.embeddings import Embeddings

from custom_utils.image_embedder import get_image_embedding_from_frame

logger = logging.getLogger(__name__)


class ImageEmbeddingFunction(Embeddings):
    """
    Custom embedding function for Chroma that uses Azure Computer Vision.

    This class wraps our custom embedder to work with Chroma's expected interface.
    It ONLY accepts numpy array images and will fail hard for any other input type.

    Implements the LangChain Embeddings interface required by Chroma.
    """

    def __init__(self):
        """Initialize the embedding function."""
        self.dimension = 1024  # Azure vision embeddings are 1024-dimensional
        logger.info("Initialized ImageEmbeddingFunction for Chroma")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents.

        Note: This method is called by Chroma, but we expect numpy arrays not text.
        Since Chroma passes texts, we'll treat this as a pass-through that returns
        zero vectors (actual embedding generation happens elsewhere).

        Args:
            texts: List of texts (ignored for image embeddings)

        Returns:
            List of zero vectors (placeholder)
        """
        # Return zero vectors for text inputs
        # The actual image embeddings are generated during storage
        logger.warning("embed_documents called with text inputs, returning zero vectors")
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Note: This method is called by Chroma for queries, but we expect numpy arrays.
        Returns a zero vector as placeholder.

        Args:
            text: Query text (ignored for image embeddings)

        Returns:
            Zero vector (placeholder)
        """
        logger.warning("embed_query called with text input, returning zero vector")
        return [0.0] * self.dimension

    def __call__(self, input_data: List[Any]) -> List[List[float]]:
        """
        Generate embeddings for input data.

        Args:
            input_data: List of numpy arrays (images) to embed

        Returns:
            List of embeddings as lists of floats

        Raises:
            TypeError: If input is not a numpy array (fails hard as required)
        """
        embeddings = []

        for i, item in enumerate(input_data):
            # Fail hard if not a numpy array
            if not isinstance(item, np.ndarray):
                raise TypeError(
                    f"ImageEmbeddingFunction only accepts numpy arrays (images). "
                    f"Got {type(item)} for item {i}. "
                    f"This embedding function is strictly for image embeddings only."
                )

            # Generate embedding from image
            embedding = get_image_embedding_from_frame(item, f"item_{i}")

            if embedding is not None:
                embeddings.append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            else:
                logger.warning(f"Failed to generate embedding for item {i}, using zero vector")
                embeddings.append([0.0] * self.dimension)

        return embeddings
