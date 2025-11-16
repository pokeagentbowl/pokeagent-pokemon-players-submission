#!/usr/bin/env python3
"""
Unit tests for Planner module

Tests the high-level strategic planner with unified VLM call.
Uses mocks to avoid actual VLM API calls and memory retrieval.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from custom_agent.mvp_hierarchical.modules.planner import (
    Planner,
    NavigationPlan,
    OtherExecutorPlan,
    PlanDecisionResponse,
    PlanResult,
    PLANNER_PROMPT
)
from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_utils.navigation_targets import NavigationTarget
from custom_utils.object_detector import DetectedObject


class TestPlanner:
    """Test suite for Planner."""

    @pytest.fixture
    def mock_reasoner(self):
        """Create a mock LangChainVLM reasoner."""
        mock = Mock()
        mock.call_vlm = Mock()
        return mock

    @pytest.fixture
    def mock_memory(self):
        """Create a mock EpisodicMemory."""
        mock = Mock()
        mock.retrieve = Mock(return_value=[])  # Default: no memories
        return mock

    @pytest.fixture
    def planner(self, mock_reasoner, mock_memory):
        """Create a Planner instance with mocks."""
        return Planner(reasoner=mock_reasoner, memory=mock_memory)

    @pytest.fixture
    def mock_perception_with_targets(self):
        """Create a mock PerceptionResult with navigation targets."""
        # Create mock detected objects
        obj1 = Mock(spec=DetectedObject)
        obj1.name = "NPC"
        obj1.entity_type = "person"

        obj2 = Mock(spec=DetectedObject)
        obj2.name = "Building"
        obj2.entity_type = "structure"

        # Create navigation targets
        target1 = NavigationTarget(
            id="object_0",
            type="object",
            map_tile_position=(10, 15),
            local_tile_position=(5, 6),
            description="NPC at local tile (5, 6)",
            entity_type="person",
            source_map_location="Route 101"
        )

        target2 = NavigationTarget(
            id="boundary_0",
            type="boundary",
            map_tile_position=(12, 20),
            local_tile_position=(14, 12),
            description="East boundary exit (length 3) at local tile (14, 12)",
            entity_type=None,
            source_map_location="Route 101"
        )

        mock = Mock(spec=PerceptionResult)
        mock.detected_objects = [obj1, obj2]
        mock.scene_description = "Route with NPC and buildings"
        mock.navigation_targets = [target1, target2]

        return mock

    @pytest.fixture
    def mock_perception_no_targets(self):
        """Create a mock PerceptionResult without navigation targets."""
        obj1 = Mock(spec=DetectedObject)
        obj1.name = "Wall"

        mock = Mock(spec=PerceptionResult)
        mock.detected_objects = [obj1]
        mock.scene_description = "Indoor area with walls"
        mock.navigation_targets = []

        return mock

    @pytest.fixture
    def sample_state_data(self):
        """Create sample game state data."""
        frame = np.zeros((160, 240, 3), dtype=np.uint8)  # Sample frame
        return {
            'frame': frame,
            'player': {'location': 'Route 101'}
        }

    @pytest.fixture
    def sample_recent_logs(self):
        """Create sample recent logs."""
        return [
            "Started game",
            "Talked to Professor Birch",
            "Received starter Pokemon",
            "Exited lab"
        ]

    # Test initialization
    def test_initialization(self, planner, mock_reasoner, mock_memory):
        """Test that Planner initializes correctly."""
        assert planner is not None
        assert planner.reasoner is mock_reasoner
        assert planner.memory is mock_memory

    # Test navigation plan (VLM chooses navigation with valid target)
    def test_create_plan_navigation_valid_target(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test creating a navigation plan with valid target index."""
        # Mock VLM decision: navigation to target index 0
        mock_decision = NavigationPlan(
            reasoning="Should talk to the NPC to progress the story",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify result
        assert isinstance(result, PlanResult)
        assert result.executor_type == 'navigation'
        assert isinstance(result.goal, dict)
        assert result.goal['target'] == mock_perception_with_targets.navigation_targets[0]
        assert "NPC at local tile (5, 6)" in result.goal['description']
        assert result.reasoning == "Should talk to the NPC to progress the story"

        # Verify VLM was called with correct parameters
        mock_reasoner.call_vlm.assert_called_once()
        call_args = mock_reasoner.call_vlm.call_args
        assert call_args.kwargs['module_name'] == "PLANNER"
        assert "AVAILABLE NAVIGATION TARGETS:" in call_args.kwargs['prompt']
        assert "0. NPC at local tile (5, 6)" in call_args.kwargs['prompt']

    # Test navigation plan with out-of-bounds target index (fallback)
    def test_create_plan_navigation_invalid_target_index(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test navigation plan with invalid target index falls back to first target."""
        # Mock VLM decision: navigation with out-of-bounds index
        mock_decision = NavigationPlan(
            reasoning="Should explore the boundary",
            executor_type='navigation',
            target_index=99  # Out of bounds
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify fallback to first target
        assert result.executor_type == 'navigation'
        assert result.goal['target'] == mock_perception_with_targets.navigation_targets[0]

    # Test navigation plan when no targets available
    def test_create_plan_navigation_no_targets(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_no_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test navigation plan when no targets are available (graceful degradation)."""
        # Mock VLM decision: navigation (shouldn't happen, but handle gracefully)
        mock_decision = NavigationPlan(
            reasoning="Should explore",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_no_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify graceful handling
        assert result.executor_type == 'navigation'
        assert result.goal['target'] is None
        assert "no targets available" in result.goal['description'].lower()

    # Test battle plan
    def test_create_plan_battle(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test creating a battle plan."""
        # Mock VLM decision: battle
        mock_decision = OtherExecutorPlan(
            reasoning="Wild Pokemon appeared, need to battle",
            executor_type='battle',
            goal="Defeat the wild Pokemon"
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify result
        assert isinstance(result, PlanResult)
        assert result.executor_type == 'battle'
        assert result.goal == "Defeat the wild Pokemon"
        assert result.reasoning == "Wild Pokemon appeared, need to battle"

    # Test general plan
    def test_create_plan_general(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_no_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test creating a general executor plan."""
        # Mock VLM decision: general
        mock_decision = OtherExecutorPlan(
            reasoning="Need to open the start menu and save the game",
            executor_type='general',
            goal="Open start menu and save"
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_no_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify result
        assert result.executor_type == 'general'
        assert result.goal == "Open start menu and save"
        assert result.reasoning == "Need to open the start menu and save the game"

    # Test memory retrieval integration
    def test_create_plan_with_memories(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test that planner retrieves and uses memories."""
        # Mock memory entries
        mock_mem1 = Mock()
        mock_mem1.step_number = 10
        mock_mem1.perception = Mock()
        mock_mem1.perception.detected_objects = [Mock(), Mock()]

        mock_mem2 = Mock()
        mock_mem2.step_number = 15
        mock_mem2.perception = Mock()
        mock_mem2.perception.detected_objects = [Mock()]

        mock_memory.retrieve.return_value = [mock_mem1, mock_mem2]

        # Mock VLM decision
        mock_decision = NavigationPlan(
            reasoning="Based on past experience, should talk to NPC",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify memory retrieval was called
        mock_memory.retrieve.assert_called_once()
        call_args = mock_memory.retrieve.call_args
        assert 'query_image' in call_args.kwargs
        assert call_args.kwargs['top_k'] == 3

        # Verify memories were included in prompt
        prompt = mock_reasoner.call_vlm.call_args.kwargs['prompt']
        assert "RELEVANT PAST EXPERIENCES:" in prompt
        assert "Step 10: 2 objects detected" in prompt
        assert "Step 15: 1 objects detected" in prompt

    # Test prompt formatting with no recent logs
    def test_create_plan_no_recent_logs(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data
    ):
        """Test prompt formatting when no recent logs are available."""
        mock_decision = NavigationPlan(
            reasoning="First action",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan with empty logs
        result = planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=[]
        )

        # Verify prompt includes "No recent actions"
        prompt = mock_reasoner.call_vlm.call_args.kwargs['prompt']
        assert "No recent actions" in prompt

    # Test prompt formatting with no detected objects
    def test_create_plan_no_detected_objects(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        sample_state_data,
        sample_recent_logs
    ):
        """Test prompt formatting when no objects are detected."""
        # Create perception with no objects
        mock_perception = Mock(spec=PerceptionResult)
        mock_perception.detected_objects = []
        mock_perception.scene_description = "Empty area"
        mock_perception.navigation_targets = []

        mock_decision = OtherExecutorPlan(
            reasoning="Nothing to interact with",
            executor_type='general',
            goal="Explore"
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        result = planner.create_plan(
            perception=mock_perception,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify prompt includes "No objects detected"
        prompt = mock_reasoner.call_vlm.call_args.kwargs['prompt']
        assert "No objects detected" in prompt

    # Test VLM call parameters
    def test_vlm_call_parameters(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data,
        sample_recent_logs
    ):
        """Test that VLM is called with correct parameters."""
        mock_decision = NavigationPlan(
            reasoning="Test",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=sample_recent_logs
        )

        # Verify call parameters
        mock_reasoner.call_vlm.assert_called_once()
        call_kwargs = mock_reasoner.call_vlm.call_args.kwargs

        assert 'prompt' in call_kwargs
        assert 'image' in call_kwargs
        assert call_kwargs['module_name'] == "PLANNER"
        assert 'structured_output_model' in call_kwargs

        # Verify image is numpy array
        assert isinstance(call_kwargs['image'], np.ndarray)

    # Test recent logs truncation (only last 10)
    def test_recent_logs_truncation(
        self,
        planner,
        mock_reasoner,
        mock_memory,
        mock_perception_with_targets,
        sample_state_data
    ):
        """Test that only last 10 recent logs are included in prompt."""
        # Create 15 logs
        many_logs = [f"Action {i}" for i in range(15)]

        mock_decision = NavigationPlan(
            reasoning="Test",
            executor_type='navigation',
            target_index=0
        )
        mock_reasoner.call_vlm.return_value = PlanDecisionResponse(decision=mock_decision)

        # Create plan
        planner.create_plan(
            perception=mock_perception_with_targets,
            state_data=sample_state_data,
            recent_logs=many_logs
        )

        # Verify prompt only includes last 10 logs
        prompt = mock_reasoner.call_vlm.call_args.kwargs['prompt']
        assert "Action 5" in prompt  # Should include actions 5-14 (last 10)
        assert "Action 4" not in prompt  # Should not include action 4 (11th from end)
