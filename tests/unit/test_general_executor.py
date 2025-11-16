#!/usr/bin/env python3
"""
Unit tests for GeneralExecutor

Tests the VLM-based general executor for menus, dialogues, and interactions.
Uses mocks to avoid actual VLM API calls.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from custom_agent.mvp_hierarchical.executors.general_executor import (
    GeneralExecutor,
    GeneralExecutorDecision
)
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult


class TestGeneralExecutor:
    """Test suite for GeneralExecutor."""

    @pytest.fixture
    def mock_reasoner(self):
        """Create a mock LangChainVLM reasoner."""
        mock = Mock()
        mock.call_vlm = Mock()
        return mock

    @pytest.fixture
    def executor(self, mock_reasoner):
        """Create a GeneralExecutor instance with mock reasoner."""
        return GeneralExecutor(reasoner=mock_reasoner)

    @pytest.fixture
    def mock_perception(self):
        """Create a mock PerceptionResult."""
        mock = Mock()
        mock.detected_objects = []
        mock.scene_description = "Player is in a town with NPCs nearby"
        mock.scene_embedding = None
        return mock

    def test_initialization(self, executor, mock_reasoner):
        """Test that GeneralExecutor initializes correctly."""
        assert executor is not None
        assert executor.reasoner is mock_reasoner
        assert executor.internal_state['actions_taken'] == 0
        assert executor.internal_state['max_actions'] == 20

    def test_execute_step_in_progress(self, executor, mock_reasoner, mock_perception):
        """Test execute_step returns buttons when VLM says in_progress."""
        # Mock VLM decision: in_progress
        mock_decision = GeneralExecutorDecision(
            reasoning="Need to advance dialogue with NPC",
            status='in_progress',
            buttons=['A'],
            summary=None
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {
                'menu_open': False,
                'text_box_active': True,
                'in_battle': False
            },
            'player': {'location': 'Littleroot Town'}
        }
        goal = "Talk to Professor Birch"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert isinstance(result, ExecutorResult)
        assert result.actions == ['A']
        assert result.status == 'in_progress'
        assert result.summary is None
        assert executor.internal_state['actions_taken'] == 1

    def test_execute_step_completed(self, executor, mock_reasoner, mock_perception):
        """Test execute_step returns completed when VLM says completed."""
        # Mock VLM decision: completed
        mock_decision = GeneralExecutorDecision(
            reasoning="Dialogue finished, text box closed",
            status='completed',
            buttons=[],
            summary="Successfully talked to Professor Birch"
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {
                'menu_open': False,
                'text_box_active': False,
                'in_battle': False
            },
            'player': {'location': 'Littleroot Town'}
        }
        goal = "Talk to Professor Birch"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert result.actions == []
        assert result.status == 'completed'
        assert result.summary == "Successfully talked to Professor Birch"
        assert executor.internal_state['actions_taken'] == 0  # Reset

    def test_execute_step_failed(self, executor, mock_reasoner, mock_perception):
        """Test execute_step returns failed when VLM says failed."""
        # Mock VLM decision: failed
        mock_decision = GeneralExecutorDecision(
            reasoning="Battle started unexpectedly",
            status='failed',
            buttons=[],
            summary="Interrupted by wild Pokemon battle"
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {
                'menu_open': False,
                'text_box_active': False,
                'in_battle': True
            },
            'player': {'location': 'Route 101'}
        }
        goal = "Walk to Oldale Town"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert result.actions == []
        assert result.status == 'failed'
        assert result.summary == "Interrupted by wild Pokemon battle"
        assert executor.internal_state['actions_taken'] == 0  # Reset

    def test_execute_step_multiple_buttons(self, executor, mock_reasoner, mock_perception):
        """Test execute_step with multiple button sequence."""
        # Mock VLM decision with multiple buttons
        mock_decision = GeneralExecutorDecision(
            reasoning="Need to navigate menu down then select",
            status='in_progress',
            buttons=['DOWN', 'DOWN', 'A'],
            summary=None
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {
                'menu_open': True,
                'text_box_active': False,
                'in_battle': False
            },
            'player': {'location': 'Littleroot Town'}
        }
        goal = "Open menu and save game"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert result.actions == ['DOWN', 'DOWN', 'A']
        assert result.status == 'in_progress'

    def test_execute_step_max_actions_limit(self, executor, mock_reasoner, mock_perception):
        """Test safety limit prevents infinite loops."""
        # Mock VLM to always return in_progress
        mock_decision = GeneralExecutorDecision(
            reasoning="Keep trying",
            status='in_progress',
            buttons=['A'],
            summary=None
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {'menu_open': False, 'text_box_active': False, 'in_battle': False},
            'player': {'location': 'Littleroot Town'}
        }
        goal = "Talk to NPC"

        # Execute until max_actions reached
        for i in range(20):
            result = executor.execute_step(mock_perception, state_data, goal)
            assert result.status == 'in_progress'
            assert executor.internal_state['actions_taken'] == i + 1

        # Next call should fail due to safety limit
        result = executor.execute_step(mock_perception, state_data, goal)
        assert result.status == 'failed'
        assert result.actions == []
        assert 'Safety limit' in result.summary
        assert 'max actions (20)' in result.summary

    def test_is_still_valid_always_true(self, executor, mock_perception):
        """Test is_still_valid always returns True (MVP behavior)."""
        # Various state scenarios - should always be valid
        state_scenarios = [
            {'game': {'in_battle': True}},
            {'game': {'in_battle': False}},
            {'game': {'menu_open': True}},
            {'game': {'text_box_active': True}},
            {},
        ]

        for state_data in state_scenarios:
            assert executor.is_still_valid(state_data, mock_perception) == True

    def test_get_state(self, executor):
        """Test get_state returns internal state."""
        executor.internal_state['actions_taken'] = 5

        state = executor.get_state()

        assert isinstance(state, dict)
        assert state['actions_taken'] == 5
        assert state['max_actions'] == 20
        # State should be a copy
        state['actions_taken'] = 10
        assert executor.internal_state['actions_taken'] == 5

    def test_restore_state(self, executor):
        """Test restore_state sets internal state."""
        new_state = {
            'actions_taken': 7,
            'max_actions': 15
        }

        executor.restore_state(new_state)

        assert executor.internal_state['actions_taken'] == 7
        assert executor.internal_state['max_actions'] == 15
        # Should be a copy
        new_state['actions_taken'] = 20
        assert executor.internal_state['actions_taken'] == 7

    def test_vlm_call_parameters(self, executor, mock_reasoner, mock_perception):
        """Test that VLM is called with correct parameters."""
        mock_decision = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {
                'menu_open': False,
                'text_box_active': True,
                'in_battle': False
            },
            'player': {'location': 'Littleroot Town'}
        }
        goal = "Talk to Professor Birch"

        executor.execute_step(mock_perception, state_data, goal)

        # Verify VLM was called
        mock_reasoner.call_vlm.assert_called_once()
        call_kwargs = mock_reasoner.call_vlm.call_args[1]

        # Check parameters
        assert 'prompt' in call_kwargs
        assert goal in call_kwargs['prompt']
        assert mock_perception.scene_description in call_kwargs['prompt']
        assert 'Littleroot Town' in call_kwargs['prompt']
        assert call_kwargs['image'] is None  # MVP doesn't pass frame
        assert call_kwargs['module_name'] == 'GENERAL_EXECUTOR'
        assert call_kwargs['structured_output_model'] == GeneralExecutorDecision

    def test_action_counter_increments(self, executor, mock_reasoner, mock_perception):
        """Test that action counter increments on each step."""
        mock_decision = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        state_data = {
            'game': {'menu_open': False, 'text_box_active': False, 'in_battle': False},
            'player': {'location': 'Test'}
        }

        assert executor.internal_state['actions_taken'] == 0

        executor.execute_step(mock_perception, state_data, "Test goal 1")
        assert executor.internal_state['actions_taken'] == 1

        executor.execute_step(mock_perception, state_data, "Test goal 2")
        assert executor.internal_state['actions_taken'] == 2

        executor.execute_step(mock_perception, state_data, "Test goal 3")
        assert executor.internal_state['actions_taken'] == 3

    def test_action_counter_resets_on_completion(self, executor, mock_reasoner, mock_perception):
        """Test that action counter resets when task completes."""
        # First, increment counter
        mock_decision_progress = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision_progress

        state_data = {
            'game': {'menu_open': False, 'text_box_active': False, 'in_battle': False},
            'player': {'location': 'Test'}
        }

        for _ in range(5):
            executor.execute_step(mock_perception, state_data, "Test")

        assert executor.internal_state['actions_taken'] == 5

        # Now complete
        mock_decision_complete = GeneralExecutorDecision(
            reasoning="Done", status='completed', buttons=[], summary="Completed"
        )
        mock_reasoner.call_vlm.return_value = mock_decision_complete

        result = executor.execute_step(mock_perception, state_data, "Test")
        assert result.status == 'completed'
        assert executor.internal_state['actions_taken'] == 0

    def test_action_counter_resets_on_failure(self, executor, mock_reasoner, mock_perception):
        """Test that action counter resets when task fails."""
        # First, increment counter
        mock_decision_progress = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision_progress

        state_data = {
            'game': {'menu_open': False, 'text_box_active': False, 'in_battle': False},
            'player': {'location': 'Test'}
        }

        for _ in range(5):
            executor.execute_step(mock_perception, state_data, "Test")

        assert executor.internal_state['actions_taken'] == 5

        # Now fail
        mock_decision_failed = GeneralExecutorDecision(
            reasoning="Failed", status='failed', buttons=[], summary="Task failed"
        )
        mock_reasoner.call_vlm.return_value = mock_decision_failed

        result = executor.execute_step(mock_perception, state_data, "Test")
        assert result.status == 'failed'
        assert executor.internal_state['actions_taken'] == 0

    def test_executor_protocol_compliance(self, executor):
        """Test that GeneralExecutor has all required protocol methods."""
        # Check all ExecutorProtocol methods exist
        assert hasattr(executor, 'execute_step')
        assert callable(executor.execute_step)

        assert hasattr(executor, 'is_still_valid')
        assert callable(executor.is_still_valid)

        assert hasattr(executor, 'get_state')
        assert callable(executor.get_state)

        assert hasattr(executor, 'restore_state')
        assert callable(executor.restore_state)

    def test_missing_game_state(self, executor, mock_reasoner, mock_perception):
        """Test executor handles missing game state gracefully."""
        mock_decision = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        # Missing game state
        state_data = {'player': {'location': 'Test'}}

        result = executor.execute_step(mock_perception, state_data, "Test goal")

        # Should not crash
        assert result.status == 'in_progress'
        # Verify VLM was called
        mock_reasoner.call_vlm.assert_called_once()

    def test_missing_player_location(self, executor, mock_reasoner, mock_perception):
        """Test executor handles missing player location gracefully."""
        mock_decision = GeneralExecutorDecision(
            reasoning="Test", status='in_progress', buttons=['A']
        )
        mock_reasoner.call_vlm.return_value = mock_decision

        # Missing player location
        state_data = {'game': {'menu_open': False}}

        result = executor.execute_step(mock_perception, state_data, "Test goal")

        # Should not crash
        assert result.status == 'in_progress'
        # Verify VLM was called with 'unknown' for location
        prompt = mock_reasoner.call_vlm.call_args[1]['prompt']
        assert 'Location: unknown' in prompt


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
