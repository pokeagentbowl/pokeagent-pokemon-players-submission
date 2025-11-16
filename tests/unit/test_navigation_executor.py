#!/usr/bin/env python3
"""
Unit tests for NavigationExecutor

Tests the navigation executor that executes paths to chosen targets.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from custom_agent.mvp_hierarchical.executors.navigation_executor import NavigationExecutor
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult
from custom_utils.navigation_targets import NavigationTarget


class TestNavigationExecutor:
    """Test suite for NavigationExecutor."""

    @pytest.fixture
    def executor(self):
        """Create a NavigationExecutor instance for testing."""
        return NavigationExecutor(action_batch_size=3, stuck_threshold=5)

    @pytest.fixture
    def mock_perception(self):
        """Create a mock PerceptionResult."""
        mock = Mock()
        mock.detected_objects = []
        mock.scene_description = "Game scene"
        mock.scene_embedding = None
        return mock

    @pytest.fixture
    def sample_state_data(self):
        """Create sample state data."""
        return {
            'player': {
                'position': {'x': 10, 'y': 15},
                'location': 'PALLET_TOWN'
            },
            'game': {
                'is_in_battle': False,
                'game_state': 'overworld'
            },
            'map': {
                'player_centered_grid': [['.' for _ in range(15)] for _ in range(15)]
            }
        }

    @pytest.fixture
    def sample_navigation_target(self):
        """Create a sample NavigationTarget."""
        return NavigationTarget(
            id='object_0',
            type='object',
            description='Test NPC',
            local_tile_position=(8, 7),
            map_tile_position=(11, 15),
            source_map_location='PALLET_TOWN'
        )

    def test_initialization(self, executor):
        """Test that NavigationExecutor initializes correctly."""
        assert executor is not None
        assert executor.action_batch_size == 3
        assert executor.stuck_threshold == 5
        assert executor.movement_mode == "naive"
        assert executor.internal_state['phase'] == 'idle'
        assert executor.pathfinder is not None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        executor = NavigationExecutor(
            action_batch_size=5,
            stuck_threshold=10,
            movement_mode="facing_aware"
        )
        assert executor.action_batch_size == 5
        assert executor.stuck_threshold == 10
        assert executor.movement_mode == "facing_aware"

    def test_update_state_valid(self, executor, sample_state_data):
        """Test state extraction from valid state data."""
        executor._update_state(sample_state_data)

        assert executor.player_map_tile_x == 10
        assert executor.player_map_tile_y == 15
        assert executor.player_map == 'PALLET_TOWN'
        assert executor.traversability_map is not None

    def test_update_state_missing_position(self, executor):
        """Test state extraction with missing position."""
        state_data = {
            'player': {},
            'game': {'is_in_battle': False, 'game_state': 'overworld'},
            'map': {}
        }
        executor._update_state(state_data)

        assert executor.player_map_tile_x is None
        assert executor.player_map_tile_y is None

    def test_update_state_missing_location(self, executor, sample_state_data):
        """Test state extraction with missing location."""
        state_data = sample_state_data.copy()
        state_data['player'] = {'position': {'x': 10, 'y': 15}}
        executor._update_state(state_data)

        assert executor.player_map_tile_x == 10
        assert executor.player_map_tile_y == 15
        assert executor.player_map is None

    def test_is_still_valid_in_overworld(self, executor, mock_perception):
        """Test is_still_valid returns True when in overworld."""
        state_data = {
            'game': {
                'is_in_battle': False,
                'game_state': 'overworld'
            }
        }

        assert executor.is_still_valid(state_data, mock_perception) == True

    def test_is_still_valid_in_battle(self, executor, mock_perception):
        """Test is_still_valid returns False when in battle."""
        state_data = {
            'game': {
                'is_in_battle': True,
                'game_state': 'overworld'
            }
        }

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_is_still_valid_not_overworld(self, executor, mock_perception):
        """Test is_still_valid returns False when not in overworld."""
        state_data = {
            'game': {
                'is_in_battle': False,
                'game_state': 'menu'
            }
        }

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_is_still_valid_missing_game_state(self, executor, mock_perception):
        """Test is_still_valid handles missing game state."""
        state_data = {'game': {}}

        assert executor.is_still_valid(state_data, mock_perception) == False

    def test_execute_step_invalid_goal_format(self, executor, mock_perception, sample_state_data):
        """Test execute_step with invalid goal format."""
        invalid_goal = "just a string"

        result = executor.execute_step(mock_perception, sample_state_data, invalid_goal)

        assert result.status == 'failed'
        assert "Invalid goal format" in result.summary
        assert result.actions == []

    def test_execute_step_goal_as_navigation_target(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test execute_step accepts NavigationTarget directly as goal."""
        # Mock pathfinding to return a simple path
        with patch.object(executor, '_plan_path_to_target', return_value=['UP', 'UP', 'RIGHT']):
            result = executor.execute_step(
                mock_perception, sample_state_data, sample_navigation_target
            )

            assert result.status == 'in_progress'
            assert len(result.actions) <= executor.action_batch_size
            assert executor.internal_state['phase'] == 'navigating'

    def test_execute_step_goal_as_dict(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test execute_step accepts dict with 'target' key."""
        goal_dict = {'target': sample_navigation_target, 'description': 'Navigate to NPC'}

        with patch.object(executor, '_plan_path_to_target', return_value=['UP', 'UP', 'RIGHT']):
            result = executor.execute_step(mock_perception, sample_state_data, goal_dict)

            assert result.status == 'in_progress'
            assert len(result.actions) <= executor.action_batch_size
            assert executor.internal_state['phase'] == 'navigating'

    def test_execute_step_no_path_found(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test execute_step when pathfinding fails."""
        with patch.object(executor, '_plan_path_to_target', return_value=[]):
            result = executor.execute_step(
                mock_perception, sample_state_data, sample_navigation_target
            )

            assert result.status == 'failed'
            assert "No path found" in result.summary
            assert result.actions == []

    def test_path_execution_batch_emission(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test that path is executed in batches."""
        long_path = ['UP'] * 10

        with patch.object(executor, '_plan_path_to_target', return_value=long_path):
            # Start navigation
            result1 = executor.execute_step(
                mock_perception, sample_state_data, sample_navigation_target
            )

            # Should emit first batch (3 actions)
            assert result1.status == 'in_progress'
            assert len(result1.actions) == 3
            assert executor.internal_state['path_index'] == 3

    def test_stuck_detection(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test stuck detection when no progress is made."""
        # Make path long enough to not exhaust before stuck detection (batch_size=3, threshold=5, so need >15 actions)
        with patch.object(executor, '_plan_path_to_target', return_value=['UP'] * 20):
            with patch.object(executor, '_has_reached_target', return_value=False):
                # Start navigation
                executor.execute_step(mock_perception, sample_state_data, sample_navigation_target)

                # Execute multiple times without position change (stuck)
                # Note: Pass dummy goal param, but executor is in 'navigating' phase so it won't use it
                for i in range(executor.stuck_threshold):
                    result = executor.execute_step(mock_perception, sample_state_data, None)

                # Should detect stuck and fail
                assert result.status == 'failed'
                assert "Stuck" in result.summary
                assert executor.internal_state['phase'] == 'idle'

    def test_reached_target_object(self, executor, sample_state_data):
        """Test target reached detection for object type."""
        target = NavigationTarget(
            id='object_0',
            type='object',
            description='NPC',
            local_tile_position=(8, 7),
            map_tile_position=(11, 15),  # Adjacent to player at (10, 15)
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        # Player at (10, 15), target at (11, 15) -> distance = 1
        assert executor._has_reached_target(sample_state_data) == True

    def test_not_reached_target_object(self, executor, sample_state_data):
        """Test target not reached for object type when too far."""
        target = NavigationTarget(
            id='object_1',
            type='object',
            description='NPC',
            local_tile_position=(8, 7),
            map_tile_position=(13, 15),  # Distance = 3 from player
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        assert executor._has_reached_target(sample_state_data) == False

    def test_reached_target_warp(self, executor, sample_state_data):
        """Test target reached detection for warp type (map changed)."""
        target = NavigationTarget(
            id='warp_0',
            type='warp',
            description='Door',
            local_tile_position=(8, 7),
            map_tile_position=(10, 15),
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        # Change map to simulate warp
        state_data_after_warp = sample_state_data.copy()
        state_data_after_warp['player'] = {
            'position': {'x': 5, 'y': 5},
            'location': 'INSIDE_HOUSE'
        }

        # Update state to reflect new map before checking
        executor._update_state(state_data_after_warp)

        assert executor._has_reached_target(state_data_after_warp) == True

    def test_not_reached_target_warp_same_map(self, executor, sample_state_data):
        """Test warp target not reached when map hasn't changed."""
        target = NavigationTarget(
            id='warp_1',
            type='warp',
            description='Door',
            local_tile_position=(8, 7),
            map_tile_position=(10, 15),
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        # Same map - warp not triggered yet
        assert executor._has_reached_target(sample_state_data) == False

    def test_reached_target_boundary_map_changed(self, executor, sample_state_data):
        """Test boundary target reached when map changes."""
        target = NavigationTarget(
            id='boundary_0',
            type='boundary',
            description='Map edge',
            local_tile_position=(14, 7),
            map_tile_position=(20, 15),
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        # Change map
        state_data_new_map = sample_state_data.copy()
        state_data_new_map['player'] = {
            'position': {'x': 1, 'y': 15},
            'location': 'ROUTE_1'
        }

        # Update state to reflect new map before checking
        executor._update_state(state_data_new_map)

        assert executor._has_reached_target(state_data_new_map) == True

    def test_reached_target_boundary_on_tile(self, executor, sample_state_data):
        """Test boundary target reached when on boundary tile."""
        target = NavigationTarget(
            id='boundary_1',
            type='boundary',
            description='Map edge',
            local_tile_position=(14, 7),
            map_tile_position=(10, 15),  # Same as player position
            source_map_location='PALLET_TOWN'
        )

        executor.internal_state['current_target'] = target
        executor._update_state(sample_state_data)

        # Distance = 0, same map
        assert executor._has_reached_target(sample_state_data) == True

    def test_path_exhaustion(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test handling when path is exhausted without reaching target."""
        short_path = ['UP', 'RIGHT']

        with patch.object(executor, '_plan_path_to_target', return_value=short_path):
            with patch.object(executor, '_has_reached_target', return_value=False):
                # Start navigation
                executor.execute_step(mock_perception, sample_state_data, sample_navigation_target)

                # Simulate position change to avoid stuck detection
                state_data_moved = sample_state_data.copy()
                state_data_moved['player']['position'] = {'x': 11, 'y': 16}

                # Execute until path exhausted
                result = executor.execute_step(mock_perception, state_data_moved, None)

                # Path should be exhausted
                assert result.status == 'failed'
                assert "Path exhausted" in result.summary
                assert executor.internal_state['phase'] == 'idle'

    def test_complete_navigation_sequence(
        self, executor, mock_perception, sample_state_data, sample_navigation_target
    ):
        """Test a complete navigation sequence from start to completion."""
        path = ['UP', 'UP', 'RIGHT']

        with patch.object(executor, '_plan_path_to_target', return_value=path):
            # Start navigation
            result1 = executor.execute_step(
                mock_perception, sample_state_data, sample_navigation_target
            )

            assert result1.status == 'in_progress'
            assert executor.internal_state['phase'] == 'navigating'

            # Simulate reaching target
            with patch.object(executor, '_has_reached_target', return_value=True):
                result2 = executor.execute_step(mock_perception, sample_state_data, None)

                assert result2.status == 'completed'
                assert "Reached target" in result2.summary
                assert executor.internal_state['phase'] == 'idle'

    def test_get_state(self, executor):
        """Test get_state returns internal state."""
        executor.internal_state['phase'] = 'navigating'
        executor.internal_state['path_index'] = 5

        state = executor.get_state()

        assert isinstance(state, dict)
        assert state['phase'] == 'navigating'
        assert state['path_index'] == 5

    def test_restore_state(self, executor):
        """Test restore_state sets internal state."""
        new_state = {
            'phase': 'navigating',
            'current_target': None,
            'current_path': ['UP', 'RIGHT'],
            'path_index': 1,
            'steps_since_progress': 2,
            'last_player_pos': (10, 15),
            'last_player_map': 'PALLET_TOWN',
            'facing_verified': True,
            'awaiting_probe_result': False,
            'naive_path': []
        }

        executor.restore_state(new_state)

        assert executor.internal_state['phase'] == 'navigating'
        assert executor.internal_state['path_index'] == 1
        assert executor.internal_state['steps_since_progress'] == 2

    def test_facing_aware_mode_probe_emission(self):
        """Test that facing_aware mode emits probe action first."""
        executor = NavigationExecutor(movement_mode="facing_aware")

        sample_target = NavigationTarget(
            id='object_0',
            type='object',
            description='NPC',
            local_tile_position=(8, 7),
            map_tile_position=(11, 15),
            source_map_location='PALLET_TOWN'
        )

        state_data = {
            'player': {
                'position': {'x': 10, 'y': 15},
                'location': 'PALLET_TOWN'
            },
            'game': {
                'is_in_battle': False,
                'game_state': 'overworld'
            },
            'map': {}
        }

        mock_perception = Mock()

        with patch.object(executor, '_plan_path_to_target', return_value=['UP', 'UP', 'RIGHT']):
            result = executor.execute_step(mock_perception, state_data, sample_target)

            # Should emit only first action as probe
            assert result.status == 'in_progress'
            assert result.actions == ['UP']
            assert executor.internal_state['awaiting_probe_result'] == True

    def test_executor_protocol_compliance(self, executor):
        """Test that NavigationExecutor has all required protocol methods."""
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
