"""Episodic memory - stores and retrieves experiences using Chroma vector store."""

import numpy as np
import time
import os
import logging
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_chroma import Chroma
from chromadb.config import Settings

from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_utils.chroma_embeddings import ImageEmbeddingFunction
from custom_utils.cache_utils import create_cached_embeddings

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """Single episodic memory entry."""
    step_number: int
    raw_state: dict
    image: Any  # np.ndarray
    perception: PerceptionResult
    actions: List[str]
    llm_outputs: Dict[str, str]  # All LLM outputs from this step
    timestamp: float

    class Config:
        arbitrary_types_allowed = True


class EpisodicMemory:
    """
    Episodic memory system - stores and retrieves experiences using Chroma vector store.

    Storage: Direct upload (learn method)
    - Raw state data
    - Images
    - LLM outputs (thoughts, actions)
    - Perception results (object detection)
    - Step numbers (temporal)

    Retrieval: Similarity-based (image embeddings via Chroma)

    Uses cache-backed embeddings for efficient embedding generation.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialize episodic memory with Chroma vector store and cache-backed embeddings.

        Args:
            checkpoint_dir: Directory for storing checkpoints and vector database.
                           If None, creates a default directory with timestamp.
        """
        # Create checkpoint directory
        if checkpoint_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(
                os.getcwd(),
                ".agent_checkpoints",
                f"checkpoint_{timestamp}"
            )

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Create cache-backed embeddings with Redis fallback to LocalFileStore
        # This will try Redis first, then fall back to LocalFileStore if Redis is unavailable
        self.cached_embeddings = create_cached_embeddings()

        # Initialize Chroma vector store with persistence
        # Use ImageEmbeddingFunction for proper dimension handling (1024-dim)
        chroma_persist_dir = os.path.join(self.checkpoint_dir, "chroma_db")
        os.makedirs(chroma_persist_dir, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            collection_name="episodic_memories",
            embedding_function=ImageEmbeddingFunction(),
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )

        # Store memory entries separately for full data access
        self.memories: List[MemoryEntry] = []

        logger.info(f"Initialized EpisodicMemory with checkpoint_dir: {checkpoint_dir}")
        logger.info(f"Chroma database persist directory: {chroma_persist_dir}")

    def store(
        self,
        step_number: int,
        raw_state: dict,
        image: np.ndarray,
        perception: PerceptionResult,
        actions: List[str],
        llm_outputs: dict
    ):
        """
        Store a memory entry (learn method - direct upload).

        Args:
            step_number: Current step number
            raw_state: Full game state dict
            image: Game frame (numpy array)
            perception: PerceptionResult from perception module
            actions: Actions taken this step
            llm_outputs: All LLM outputs from this step
        """
        # Create memory entry
        entry = MemoryEntry(
            step_number=step_number,
            raw_state=raw_state,
            image=image,
            perception=perception,
            actions=actions,
            llm_outputs=llm_outputs,
            timestamp=time.time()
        )

        # Store in local memory list
        memory_index = len(self.memories)
        self.memories.append(entry)

        # Generate embedding for the image using cache-backed embeddings
        # Serialize frame for caching
        import base64
        frame_bytes = image.tobytes()
        frame_b64 = base64.b64encode(frame_bytes).decode('latin-1')

        # Use cached embeddings (will cache if not already cached)
        embedding_list = self.cached_embeddings.embed_documents([frame_b64])[0]

        # Create document text for Chroma (for display/debugging only)
        # The actual retrieval is purely embedding-based
        detected_obj_names = [obj.name for obj in perception.detected_objects]
        doc_text = f"Step {step_number}: Objects: {', '.join(detected_obj_names) if detected_obj_names else 'none'}"

        # Simplified metadata: ONLY store the key to reference self.memories
        metadata = {
            "memory_index": memory_index
        }

        # Generate unique ID for this memory
        memory_id = f"memory_{step_number}_{int(entry.timestamp)}"

        # Add to Chroma vector store with pre-computed embeddings
        self.vectorstore.add_texts(
            texts=[doc_text],
            embeddings=[embedding_list],
            metadatas=[metadata],
            ids=[memory_id]
        )

        logger.debug(f"Stored memory {memory_index} for step {step_number}")

    def retrieve(
        self,
        query_image: np.ndarray,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using image similarity.

        Uses Chroma's similarity search with image embeddings.

        Args:
            query_image: Query image as numpy array
            top_k: Number of memories to retrieve

        Returns:
            List of top-k most similar memories
        """
        if not self.memories:
            return []

        try:
            # Generate embedding for query image using cache-backed embeddings
            # This will cache the embedding for faster retrieval
            import base64
            # Serialize frame for cache key
            frame_bytes = query_image.tobytes()
            frame_b64 = base64.b64encode(frame_bytes).decode('latin-1')

            # Use cached embeddings for query (will cache if not already cached)
            query_embedding_list = self.cached_embeddings.embed_query(frame_b64)

            if not query_embedding_list:
                logger.warning("Failed to generate query embedding, returning recent memories")
                return self.get_recent_memories(top_k)

            # Use Chroma's similarity search with the embedding
            results = self.vectorstore.similarity_search_by_vector(
                embedding=query_embedding_list,
                k=top_k
            )

            # Extract memory entries from results using the memory_index key
            retrieved_memories = []
            for doc in results:
                memory_idx = doc.metadata.get("memory_index")
                if memory_idx is not None and memory_idx < len(self.memories):
                    retrieved_memories.append(self.memories[memory_idx])

            logger.debug(f"Retrieved {len(retrieved_memories)} memories for query")
            return retrieved_memories

        except Exception as e:
            logger.error(f"Error during memory retrieval: {e}")
            logger.warning("Falling back to recent memories")
            return self.get_recent_memories(top_k)

    def retrieve_with_text_cue(
        self,
        query_image: np.ndarray,
        text_cue: str,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories using both image and text similarity.

        Currently focuses on image similarity as per requirements.
        Text retrieval will be implemented later.

        Args:
            query_image: Query image as numpy array
            text_cue: Text query (e.g., scene description, goal)
            top_k: Number of memories to retrieve

        Returns:
            List of top-k most similar memories
        """
        # For now, just use image-based retrieval
        # Text-based retrieval will be added in future updates
        logger.debug(f"Text cue provided but using image-based retrieval: {text_cue}")
        return self.retrieve(query_image, top_k)

    def get_recent_memories(self, n: int = 10) -> List[MemoryEntry]:
        """Get n most recent memories (for debugging/analysis)."""
        return self.memories[-n:] if len(self.memories) >= n else self.memories

    def clear(self):
        """Clear all memories (for testing/reset)."""
        self.memories = []
        # Note: Chroma vectorstore persists to disk, so we'd need to recreate it
        # For now, just clear the in-memory list
        logger.info("Cleared episodic memories (note: vector store persists to disk)")

    def __len__(self):
        """Return the number of stored memories."""
        return len(self.memories)
