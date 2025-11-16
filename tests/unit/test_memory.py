#!/usr/bin/env python3
"""
Unit tests for EpisodicMemory module

Tests the episodic memory's ability to store and retrieve experiences using Chroma.
"""

import pytest
import os
import shutil
import numpy as np
import tempfile
from unittest.mock import Mock, patch, MagicMock

from custom_agent.mvp_hierarchical.modules.memory import (
    EpisodicMemory,
    MemoryEntry
)
from custom_utils.chroma_embeddings import ImageEmbeddingFunction
from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_utils.object_detector import DetectedObject


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_checkpoint_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embedding_function():
    """Mock the image embedding function."""
    with patch('custom_utils.image_embedder.get_image_embedding_from_frame') as mock:
        # Return a dummy 1024-dim embedding
        mock.return_value = np.random.rand(1024)
        yield mock


@pytest.fixture
def test_frame():
    """Create a simple test frame."""
    return np.zeros((160, 240, 3), dtype=np.uint8)


@pytest.fixture
def test_perception_result():
    """Create a test perception result."""
    return PerceptionResult(
        detected_objects=[
            DetectedObject(
                name="person",
                confidence=0.9,
                bbox={'x': 100, 'y': 80, 'w': 20, 'h': 30},
                center_pixel=(110, 95),
                entity_type="npc"
            )
        ],
        scene_description="Test scene",
        scene_embedding=np.random.rand(1024),
        navigation_targets=[],
        llm_outputs={}
    )


class TestImageEmbeddingFunction:
    """Test suite for ImageEmbeddingFunction."""

    def test_initialization(self):
        """Test that embedding function initializes correctly."""
        emb_func = ImageEmbeddingFunction()

        assert emb_func is not None
        assert emb_func.dimension == 1024

    def test_embed_numpy_array(self, mock_embedding_function, test_frame):
        """Test embedding a numpy array."""
        emb_func = ImageEmbeddingFunction()

        # Mock the embedding function to return a specific embedding
        test_embedding = np.random.rand(1024)
        mock_embedding_function.return_value = test_embedding

        result = emb_func([test_frame])

        assert len(result) == 1
        assert len(result[0]) == 1024
        assert isinstance(result[0], list)

    def test_embed_text_raises_error(self):
        """Test embedding text (should raise TypeError)."""
        emb_func = ImageEmbeddingFunction()

        # Should raise TypeError for non-numpy inputs
        with pytest.raises(TypeError, match="ImageEmbeddingFunction only accepts numpy arrays"):
            emb_func(["some text"])

    def test_embed_multiple_items(self, mock_embedding_function, test_frame):
        """Test embedding multiple numpy arrays."""
        emb_func = ImageEmbeddingFunction()

        test_embedding = np.random.rand(1024)
        mock_embedding_function.return_value = test_embedding

        result = emb_func([test_frame, test_frame])

        assert len(result) == 2
        assert all(len(emb) == 1024 for emb in result)


class TestEpisodicMemory:
    """Test suite for EpisodicMemory."""

    def test_initialization_with_checkpoint_dir(self, mock_embedding_function, temp_checkpoint_dir):
        """Test that memory initializes with specified checkpoint directory."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        assert memory is not None
        assert memory.checkpoint_dir == temp_checkpoint_dir
        assert os.path.exists(temp_checkpoint_dir)
        assert os.path.exists(os.path.join(temp_checkpoint_dir, ".custom_cache"))
        assert memory.vectorstore is not None
        assert len(memory.memories) == 0

    def test_initialization_without_checkpoint_dir(self, mock_embedding_function):
        """Test that memory creates default checkpoint directory."""
        memory = EpisodicMemory()

        assert memory is not None
        assert memory.checkpoint_dir is not None
        assert os.path.exists(memory.checkpoint_dir)
        assert ".agent_checkpoints" in memory.checkpoint_dir

        # Cleanup
        shutil.rmtree(memory.checkpoint_dir)

    def test_store_single_memory(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test storing a single memory entry."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        raw_state = {'test': 'state'}
        actions = ['A']
        llm_outputs = {'reasoning': 'test'}

        memory.store(
            step_number=1,
            raw_state=raw_state,
            image=test_frame,
            perception=test_perception_result,
            actions=actions,
            llm_outputs=llm_outputs
        )

        assert len(memory) == 1
        assert memory.memories[0].step_number == 1
        assert memory.memories[0].actions == actions

    def test_store_multiple_memories(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test storing multiple memory entries."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        for i in range(5):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        assert len(memory) == 5
        assert memory.memories[0].step_number == 0
        assert memory.memories[4].step_number == 4

    def test_retrieve_from_empty_memory(self, mock_embedding_function, temp_checkpoint_dir, test_frame):
        """Test that retrieve returns empty list when no memories exist."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        results = memory.retrieve(test_frame, top_k=5)

        assert results == []

    def test_retrieve_memories(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test retrieving memories by similarity."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store some memories
        for i in range(3):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        # Mock Chroma's similarity search
        mock_doc = Mock()
        mock_doc.metadata = {'memory_index': 0}
        memory.vectorstore.similarity_search_by_vector = Mock(return_value=[mock_doc])

        # Retrieve memories
        results = memory.retrieve(test_frame, top_k=2)

        # Should return at least one result
        assert len(results) >= 1
        assert isinstance(results[0], MemoryEntry)

    def test_retrieve_with_text_cue(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test retrieving memories with text cue (currently uses image-based retrieval)."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store a memory
        memory.store(
            step_number=0,
            raw_state={},
            image=test_frame,
            perception=test_perception_result,
            actions=['A'],
            llm_outputs={}
        )

        # Mock Chroma's similarity search
        mock_doc = Mock()
        mock_doc.metadata = {'memory_index': 0}
        memory.vectorstore.similarity_search_by_vector = Mock(return_value=[mock_doc])

        # Retrieve with text cue
        results = memory.retrieve_with_text_cue(test_frame, "test cue", top_k=1)

        # Should use image-based retrieval for now
        assert len(results) >= 1

    def test_get_recent_memories(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test getting recent memories."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store 10 memories
        for i in range(10):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        # Get 5 most recent
        recent = memory.get_recent_memories(5)

        assert len(recent) == 5
        assert recent[0].step_number == 5  # Steps 5-9
        assert recent[4].step_number == 9

    def test_get_recent_memories_less_than_n(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test getting recent memories when fewer than n exist."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store only 3 memories
        for i in range(3):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        # Request 10 most recent
        recent = memory.get_recent_memories(10)

        # Should return all 3
        assert len(recent) == 3

    def test_clear_memories(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test clearing all memories."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store some memories
        for i in range(3):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        assert len(memory) == 3

        # Clear memories
        memory.clear()

        assert len(memory) == 0

    def test_memory_entry_structure(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test that memory entries have correct structure."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        raw_state = {'test': 'state', 'nested': {'data': 123}}
        actions = ['A', 'B']
        llm_outputs = {'reasoning': 'test reasoning', 'action': 'test action'}

        memory.store(
            step_number=42,
            raw_state=raw_state,
            image=test_frame,
            perception=test_perception_result,
            actions=actions,
            llm_outputs=llm_outputs
        )

        entry = memory.memories[0]

        assert entry.step_number == 42
        assert entry.raw_state == raw_state
        assert np.array_equal(entry.image, test_frame)
        assert entry.perception == test_perception_result
        assert entry.actions == actions
        assert entry.llm_outputs == llm_outputs
        assert isinstance(entry.timestamp, float)
        assert entry.timestamp > 0

    def test_retrieve_fallback_on_error(self, mock_embedding_function, temp_checkpoint_dir, test_frame, test_perception_result):
        """Test that retrieve falls back to recent memories on error."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Store some memories
        for i in range(3):
            memory.store(
                step_number=i,
                raw_state={'step': i},
                image=test_frame,
                perception=test_perception_result,
                actions=[f'action_{i}'],
                llm_outputs={'step': str(i)}
            )

        # Mock Chroma to raise an exception
        memory.vectorstore.similarity_search_by_vector = Mock(side_effect=Exception("Test error"))

        # Retrieve should fall back to recent memories
        results = memory.retrieve(test_frame, top_k=2)

        # Should still return results (recent memories)
        assert len(results) > 0

    def test_memory_persistence_directory_structure(self, mock_embedding_function, temp_checkpoint_dir):
        """Test that correct directory structure is created."""
        memory = EpisodicMemory(checkpoint_dir=temp_checkpoint_dir)

        # Check directories exist
        assert os.path.exists(memory.checkpoint_dir)
        assert os.path.exists(os.path.join(memory.checkpoint_dir, ".custom_cache"))
        assert os.path.exists(os.path.join(memory.checkpoint_dir, "chroma_db"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
