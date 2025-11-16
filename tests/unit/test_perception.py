#!/usr/bin/env python3
"""
Unit tests for PerceptionModule

Tests the perception module's ability to process game state into structured perception.
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from custom_agent.mvp_hierarchical.modules.perception import PerceptionModule, PerceptionResult
from custom_utils.object_detector import DetectedObject
from custom_utils.navigation_targets import NavigationTarget


@pytest.fixture
def mock_object_detector():
    """Create a mock object detector."""
    with patch('custom_agent.mvp_hierarchical.modules.perception.CachedObjectDetector') as mock:
        yield mock


# Embedding generation removed from perception module (now handled by memory module)


@pytest.fixture
def test_frame():
    """Create a simple test frame."""
    return np.zeros((160, 240, 3), dtype=np.uint8)


@pytest.fixture
def test_game_state(test_frame):
    """Create a test game state."""
    return {
        'frame': test_frame,
        'player': {
            'position': {'x': 10, 'y': 15},
            'location': 'LITTLEROOT_TOWN'
        },
        'map': {
            'visual_map': '--- MAP: LITTLEROOT_TOWN ---\n' + '\n'.join(['.' * 15 for _ in range(15)])
        }
    }


@pytest.fixture
def sample_detected_objects():
    """Create sample detected objects."""
    return [
        DetectedObject(
            name="person",
            confidence=0.9,
            bbox={'x': 100, 'y': 80, 'w': 20, 'h': 30},
            center_pixel=(110, 95),
            entity_type="npc"
        ),
        DetectedObject(
            name="door",
            confidence=0.85,
            bbox={'x': 120, 'y': 60, 'w': 15, 'h': 25},
            center_pixel=(127, 72),
            entity_type="warp"
        )
    ]


class TestPerceptionModule:
    """Test suite for PerceptionModule."""
    
    def test_initialization(self):
        """Test that perception module initializes correctly."""
        perception = PerceptionModule()
        
        assert perception is not None
        assert perception.object_detector is not None
        assert perception.reasoner is None  # Not used in MVP
    
    def test_initialization_with_reasoner(self):
        """Test initialization with a reasoner (for future compatibility)."""
        mock_reasoner = Mock()
        perception = PerceptionModule(reasoner=mock_reasoner)
        
        assert perception.reasoner is mock_reasoner
    
    def test_process_returns_perception_result(self, mock_object_detector, test_game_state, sample_detected_objects):
        """Test that process() returns a valid PerceptionResult."""
        # Setup mock object detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = sample_detected_objects
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()
        result = perception.process(test_game_state)

        assert isinstance(result, PerceptionResult)
        assert result.detected_objects == sample_detected_objects
        assert isinstance(result.scene_description, str)
        assert isinstance(result.navigation_targets, list)
    
    def test_scene_description_is_blank(self, mock_object_detector, test_game_state):
        """Test that scene description is blank as per MVP requirements."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = []
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()
        result = perception.process(test_game_state)

        # Scene description should be blank for MVP
        assert result.scene_description == ""
        assert result.llm_outputs.get('scene_description') == ""

    def test_navigation_targets_generation(self, mock_object_detector, test_game_state, sample_detected_objects):
        """Test that navigation targets are generated from detected objects."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = sample_detected_objects
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()
        result = perception.process(test_game_state)

        # Should have navigation targets (at least from objects, possibly boundaries too)
        assert len(result.navigation_targets) > 0

        # Verify navigation targets have correct structure
        for target in result.navigation_targets:
            assert isinstance(target, NavigationTarget)
            assert hasattr(target, 'type')
            assert hasattr(target, 'map_tile_position')
            assert hasattr(target, 'local_tile_position')
            assert hasattr(target, 'description')

    def test_process_with_empty_game_state(self, mock_object_detector):
        """Test that process handles empty game state gracefully."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = []
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()

        # Empty game state
        empty_state = {
            'frame': np.zeros((160, 240, 3), dtype=np.uint8),
            'player': {},
            'map': {}
        }

        result = perception.process(empty_state)

        # Should still return valid result
        assert isinstance(result, PerceptionResult)
        assert result.detected_objects == []
        assert result.scene_description == ""
        assert isinstance(result.navigation_targets, list)

    def test_object_detection_called(self, mock_object_detector, test_game_state):
        """Test that object detector is called during processing."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = []
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()
        result = perception.process(test_game_state)

        # Verify object detector was called
        mock_detector_instance.detect_objects.assert_called_once()

        # Verify it was called with the frame
        call_args = mock_detector_instance.detect_objects.call_args[0]
        assert isinstance(call_args[0], np.ndarray)

    def test_llm_outputs_structure(self, mock_object_detector, test_game_state):
        """Test that llm_outputs dict has expected structure."""
        mock_detector_instance = Mock()
        mock_detector_instance.detect_objects.return_value = []
        mock_object_detector.return_value = mock_detector_instance

        perception = PerceptionModule()
        result = perception.process(test_game_state)

        # Check llm_outputs structure
        assert result.llm_outputs is not None
        assert isinstance(result.llm_outputs, dict)
        assert 'scene_description' in result.llm_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
