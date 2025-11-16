#!/usr/bin/env python3
"""
Unit tests for BattleExecutor

Tests the MVP battle executor that spams 'A' until battle ends.
"""

import pytest
from unittest.mock import Mock, MagicMock

from custom_agent.mvp_hierarchical.executors.battle_executor import BattleExecutor
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult


class TestBattleExecutor:
    """Test suite for BattleExecutor."""

    @pytest.fixture
    def executor(self):
        """Create a BattleExecutor instance for testing."""
        return BattleExecutor()

    @pytest.fixture
    def mock_perception(self):
        """Create a mock PerceptionResult."""
        mock = Mock()
        mock.detected_objects = []
        mock.scene_description = "Battle scene"
        mock.scene_embedding = None
        return mock

    def test_initialization(self, executor):
        """Test that BattleExecutor initializes correctly."""
        assert executor is not None
        assert executor.battle_started == False
        assert hasattr(executor, 'internal_state')

    def test_execute_step_in_battle(self, executor, mock_perception):
        """Test execute_step returns 'A' action when in battle."""
        state_data = {
            'game': {
                'in_battle': True
            }
        }
        goal = "Complete the battle"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert isinstance(result, ExecutorResult)
        assert result.actions == ['A']
        assert result.status == 'in_progress'
        assert result.summary is None

    def test_execute_step_battle_ended(self, executor, mock_perception):
        """Test execute_step returns completed when battle ends."""
        state_data = {
            'game': {
                'in_battle': False
            }
        }
        goal = "Complete the battle"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert isinstance(result, ExecutorResult)
        assert result.actions == []
        assert result.status == 'completed'
        assert result.summary == "Battle completed"

    def test_execute_step_missing_battle_flag(self, executor, mock_perception):
        """Test execute_step handles missing battle flag (treats as not in battle)."""
        state_data = {
            'game': {}
        }
        goal = "Complete the battle"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert result.status == 'completed'
        assert result.actions == []

    def test_execute_step_missing_game_state(self, executor, mock_perception):
        """Test execute_step handles missing game state dict."""
        state_data = {}
        goal = "Complete the battle"

        result = executor.execute_step(mock_perception, state_data, goal)

        assert result.status == 'completed'
        assert result.actions == []

    def test_is_still_valid_in_battle(self, executor, mock_perception):
        """Test is_still_valid returns True when in battle."""
        state_data = {
            'game': {
                'in_battle': True
            }
        }

        assert executor.is_still_valid(state_data, mock_perception) == True

    def test_is_still_valid_not_in_battle(self, executor, mock_perception):
        """Test is_still_valid returns False when battle ended."""
        state_data = {
            'game': {
                'in_battle': False
            }
        }

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_is_still_valid_missing_battle_flag(self, executor, mock_perception):
        """Test is_still_valid handles missing battle flag."""
        state_data = {
            'game': {}
        }

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_is_still_valid_missing_game_state(self, executor, mock_perception):
        """Test is_still_valid handles missing game state."""
        state_data = {}

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_get_state(self, executor):
        """Test get_state returns internal state."""
        state = executor.get_state()

        assert isinstance(state, dict)
        # State should be a copy
        state['test_key'] = 'test_value'
        assert 'test_key' not in executor.internal_state

    def test_restore_state(self, executor):
        """Test restore_state sets internal state."""
        new_state = {'battle_started': True, 'custom_field': 'value'}

        executor.restore_state(new_state)

        assert executor.internal_state == new_state
        # Should be a copy, not the same reference
        new_state['another_field'] = 'another_value'
        assert 'another_field' not in executor.internal_state

    def test_battle_sequence(self, executor, mock_perception):
        """Test a complete battle sequence: start -> in progress -> completed."""
        goal = "Complete the battle"

        # Battle starts
        state_in_battle = {'game': {'in_battle': True}}
        assert executor.is_still_valid(state_in_battle, mock_perception) == True

        # Execute multiple steps while in battle
        for _ in range(5):
            result = executor.execute_step(mock_perception, state_in_battle, goal)
            assert result.actions == ['A']
            assert result.status == 'in_progress'

        # Battle ends
        state_not_in_battle = {'game': {'in_battle': False}}
        assert executor.is_still_valid(state_not_in_battle, mock_perception) == False

        result = executor.execute_step(mock_perception, state_not_in_battle, goal)
        assert result.actions == []
        assert result.status == 'completed'
        assert result.summary == "Battle completed"

    def test_state_persistence_across_calls(self, executor, mock_perception):
        """Test that internal state persists across execute_step calls."""
        goal = "Complete the battle"
        state_data = {'game': {'in_battle': True}}

        # Modify internal state
        executor.internal_state['turn_count'] = 0

        # Execute steps and verify state persists
        for i in range(3):
            executor.internal_state['turn_count'] = i
            result = executor.execute_step(mock_perception, state_data, goal)
            assert result.status == 'in_progress'
            assert executor.internal_state['turn_count'] == i

    def test_executor_protocol_compliance(self, executor):
        """Test that BattleExecutor has all required protocol methods."""
        # Check all ExecutorProtocol methods exist
        assert hasattr(executor, 'execute_step')
        assert callable(executor.execute_step)

        assert hasattr(executor, 'is_still_valid')
        assert callable(executor.is_still_valid)

        assert hasattr(executor, 'get_state')
        assert callable(executor.get_state)

        assert hasattr(executor, 'restore_state')
        assert callable(executor.restore_state)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
