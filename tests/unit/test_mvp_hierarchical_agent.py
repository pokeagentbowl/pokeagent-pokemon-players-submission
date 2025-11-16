#!/usr/bin/env python3
"""
Unit tests for MVPHierarchicalAgent

Tests the main agent orchestrator's coordination of all modules and executors.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

from custom_agent.mvp_hierarchical.agent import MVPHierarchicalAgent, AgentState
from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_agent.mvp_hierarchical.modules.saliency import SaliencyResult
from custom_agent.mvp_hierarchical.modules.planner import PlanResult
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult
from custom_utils.object_detector import DetectedObject
from custom_utils.navigation_targets import NavigationTarget


# ============================================================================
# Fixtures
# ============================================================================

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
            'location': 'LITTLEROOT_TOWN',
            'in_battle': False,
            'in_overworld': True
        },
        'map': {
            'visual_map': '--- MAP: LITTLEROOT_TOWN ---\n' + '\n'.join(['.' * 15 for _ in range(15)])
        }
    }


@pytest.fixture
def mock_perception_result():
    """Create a mock perception result."""
    detected_objects = [
        DetectedObject(
            name='NPC',
            confidence=0.9,
            bbox={'x': 0, 'y': 0, 'w': 10, 'h': 10},
            center_pixel=(5, 5)
        )
    ]
    navigation_targets = [
        NavigationTarget(
            id='target_0',
            type='object',
            map_tile_position=(5, 5),
            local_tile_position=(5, 5),
            description='NPC at (5, 5)',
            detected_object=detected_objects[0]
        )
    ]
    return PerceptionResult(
        detected_objects=detected_objects,
        scene_description='',
        navigation_targets=navigation_targets,
        llm_outputs={'scene_description': ''}
    )


@pytest.fixture
def mock_plan_result_navigation(mock_perception_result):
    """Create a mock plan result for navigation."""
    nav_target = mock_perception_result.navigation_targets[0]
    return PlanResult(
        executor_type='navigation',
        goal={
            'target': nav_target,
            'description': 'Navigate to NPC at (5, 5)'
        },
        reasoning='Need to reach the NPC to interact'
    )


@pytest.fixture
def mock_plan_result_battle():
    """Create a mock plan result for battle."""
    return PlanResult(
        executor_type='battle',
        goal='Defeat the opponent Pokemon',
        reasoning='Currently in battle, need to win'
    )


@pytest.fixture
def mock_plan_result_general():
    """Create a mock plan result for general."""
    return PlanResult(
        executor_type='general',
        goal='Advance through dialogue',
        reasoning='NPC is talking, need to respond'
    )


@pytest.fixture
def mock_executor_result_in_progress():
    """Create a mock executor result that's in progress."""
    return ExecutorResult(
        actions=['UP', 'UP', 'RIGHT'],
        status='in_progress',
        summary=None
    )


@pytest.fixture
def mock_executor_result_completed():
    """Create a mock executor result that's completed."""
    return ExecutorResult(
        actions=['A'],
        status='completed',
        summary='Successfully reached target'
    )


@pytest.fixture
def mock_executor_result_failed():
    """Create a mock executor result that failed."""
    return ExecutorResult(
        actions=[],
        status='failed',
        summary='Failed to find path to target'
    )


# ============================================================================
# Test Class: Initialization
# ============================================================================

class TestInitialization:
    """Test agent initialization."""

    def test_initialization(self):
        """Test that agent initializes all modules and executors correctly."""
        with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory'), \
             patch('custom_agent.mvp_hierarchical.agent.LangChainVLM'), \
             patch('custom_agent.mvp_hierarchical.agent.PerceptionModule'), \
             patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector'), \
             patch('custom_agent.mvp_hierarchical.agent.Planner'), \
             patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor'), \
             patch('custom_agent.mvp_hierarchical.agent.BattleExecutor'), \
             patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor'):

            agent = MVPHierarchicalAgent(
                backend='github_models',
                model_name='gpt-4o-mini',
                temperature=0,
                action_batch_size=3
            )

            # Verify all components initialized
            assert agent.memory is not None
            assert agent.reasoner is not None
            assert agent.perception is not None
            assert agent.saliency is not None
            assert agent.planner is not None
            assert 'navigation' in agent.executors
            assert 'battle' in agent.executors
            assert 'general' in agent.executors
            assert agent.agent_state is not None
            assert agent.step_count == 0
            assert agent.current_step_llm_outputs == {}


# ============================================================================
# Test Class: Step Flow
# ============================================================================

class TestStepFlow:
    """Test the main step flow."""

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_normal_step_flow(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_executor_result_in_progress
    ):
        """Test normal step: perception → saliency → executor → memory."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='Executor still valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = mock_executor_result_in_progress
        mock_navigation_cls.return_value = mock_executor

        # Create agent and setup state
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Test goal'}

        # Execute step
        result = agent.step(test_game_state)

        # Verify flow
        mock_perception.process.assert_called_once_with(test_game_state)
        mock_saliency.check_validity.assert_called_once()
        mock_executor.execute_step.assert_called_once()
        agent.memory.store.assert_called_once()

        # Verify result
        assert result['action'] == ['UP', 'UP', 'RIGHT']
        assert 'reasoning' in result
        assert agent.step_count == 1

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_first_step_creates_plan(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_navigation,
        mock_executor_result_in_progress
    ):
        """Test first step: no executor → planner creates plan → executor runs."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_navigation

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = mock_executor_result_in_progress
        mock_navigation_cls.return_value = mock_executor

        # Create agent (no current executor)
        agent = MVPHierarchicalAgent()

        # Execute step
        result = agent.step(test_game_state)

        # Verify planner was called
        mock_planner.create_plan.assert_called_once()

        # Verify executor was assigned and executed
        assert agent.agent_state.current_executor_type == 'navigation'
        assert agent.agent_state.current_executor is not None
        mock_executor.execute_step.assert_called_once()

        # Verify result
        assert result['action'] == ['UP', 'UP', 'RIGHT']

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_executor_completion_clears_state(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_executor_result_completed
    ):
        """Test executor completion: clears state → next step triggers replan."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='Executor still valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = mock_executor_result_completed
        mock_navigation_cls.return_value = mock_executor

        # Create agent with current executor
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Test goal'}

        # Execute step
        result = agent.step(test_game_state)

        # Verify executor completed and state was cleared
        assert agent.agent_state.current_executor is None
        assert agent.agent_state.current_executor_type is None
        assert agent.agent_state.current_goal is None

        # Verify recent logs updated
        assert len(agent.agent_state.recent_logs) > 0
        assert 'Successfully reached target' in agent.agent_state.recent_logs[-1]

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_empty_actions_when_no_executor(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test that empty actions are returned when no executor."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value
        # Planner fails to create plan (edge case)
        mock_planner.create_plan.side_effect = Exception("Planning failed")

        # Create agent (no current executor)
        agent = MVPHierarchicalAgent()

        # Execute step should handle error gracefully
        with pytest.raises(Exception):
            result = agent.step(test_game_state)

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_return_format(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_executor_result_in_progress
    ):
        """Test that step returns correct format: {'action': actions, 'reasoning': str}."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='Executor still valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = mock_executor_result_in_progress
        mock_navigation_cls.return_value = mock_executor

        # Create agent with current executor
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Test goal'}

        # Execute step
        result = agent.step(test_game_state)

        # Verify return format
        assert isinstance(result, dict)
        assert 'action' in result
        assert 'reasoning' in result
        assert isinstance(result['action'], list)
        assert isinstance(result['reasoning'], str)


# ============================================================================
# Test Class: Goal Suspension/Resumption
# ============================================================================

class TestGoalSuspensionResumption:
    """Test goal suspension and resumption."""

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_suspend_goal_when_executor_invalid(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_battle
    ):
        """Test that goal is suspended when executor becomes invalid."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='NavigationExecutor became invalid'
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_battle

        mock_nav_executor = MagicMock()
        mock_nav_executor.get_state.return_value = {'phase': 'navigating', 'path': ['UP', 'UP']}
        mock_navigation_cls.return_value = mock_nav_executor

        mock_battle_executor = MagicMock()
        mock_battle_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='in_progress', summary=None
        )
        mock_battle_cls.return_value = mock_battle_executor

        # Create agent with navigation executor
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_nav_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Navigate to NPC'}

        # Execute step (executor becomes invalid)
        result = agent.step(test_game_state)

        # Verify goal was suspended
        assert len(agent.agent_state.suspended_goals) == 1
        suspended = agent.agent_state.suspended_goals[0]
        assert suspended['executor_type'] == 'navigation'
        assert suspended['goal'] == {'target': None, 'description': 'Navigate to NPC'}
        assert suspended['executor_state'] == {'phase': 'navigating', 'path': ['UP', 'UP']}

        # Verify new plan created (battle)
        assert agent.agent_state.current_executor_type == 'battle'

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_resume_goal_when_executor_completes(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test that suspended goal is resumed when executor completes."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='No current executor'
        )

        mock_nav_executor = MagicMock()
        mock_battle_executor = MagicMock()
        mock_battle_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_nav_executor
        mock_battle_cls.return_value = mock_battle_executor

        # Create agent with suspended goal
        agent = MVPHierarchicalAgent()
        agent.agent_state.suspended_goals = [{
            'executor_type': 'navigation',
            'goal': {'target': None, 'description': 'Navigate to NPC'},
            'executor_state': {'phase': 'navigating', 'path': ['UP', 'UP']}
        }]

        # Execute step (should resume)
        result = agent.step(test_game_state)

        # Verify goal was resumed
        assert len(agent.agent_state.suspended_goals) == 0
        assert agent.agent_state.current_executor_type == 'navigation'
        assert agent.agent_state.current_goal == {'target': None, 'description': 'Navigate to NPC'}

        # Verify executor state was restored
        mock_nav_executor.restore_state.assert_called_once_with(
            {'phase': 'navigating', 'path': ['UP', 'UP']}
        )

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_multiple_suspensions_stack_behavior(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls
    ):
        """Test that multiple suspensions use stack (LIFO) behavior."""
        agent = MVPHierarchicalAgent()

        # Add multiple suspended goals
        agent.agent_state.suspended_goals = [
            {'executor_type': 'navigation', 'goal': 'Goal 1', 'executor_state': {}},
            {'executor_type': 'general', 'goal': 'Goal 2', 'executor_state': {}},
            {'executor_type': 'battle', 'goal': 'Goal 3', 'executor_state': {}}
        ]

        # Resume should pop from end (LIFO)
        agent._resume_suspended_goal()
        assert agent.agent_state.current_executor_type == 'battle'
        assert len(agent.agent_state.suspended_goals) == 2

        agent._resume_suspended_goal()
        assert agent.agent_state.current_executor_type == 'general'
        assert len(agent.agent_state.suspended_goals) == 1

        agent._resume_suspended_goal()
        assert agent.agent_state.current_executor_type == 'navigation'
        assert len(agent.agent_state.suspended_goals) == 0

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_nested_suspension_navigation_battle_resume(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_battle,
        mock_executor_result_completed
    ):
        """Test nested suspension: navigation → battle interrupt → resume navigation."""
        # This is a more complex integration-style test, but kept as unit test
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_battle

        mock_nav_executor = MagicMock()
        mock_nav_executor.get_state.return_value = {'phase': 'navigating'}
        mock_nav_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )

        mock_battle_executor = MagicMock()
        # First call returns in_progress, second call returns completed
        mock_battle_executor.execute_step.side_effect = [
            ExecutorResult(actions=['A'], status='in_progress', summary=None),
            mock_executor_result_completed
        ]

        mock_navigation_cls.return_value = mock_nav_executor
        mock_battle_cls.return_value = mock_battle_executor

        # Create agent
        agent = MVPHierarchicalAgent()

        # Step 1: Navigation in progress
        agent.agent_state.current_executor = mock_nav_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Navigate to NPC'}

        # Step 2: Battle interrupts (executor becomes invalid)
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='Entered battle'
        )
        agent.step(test_game_state)

        # Verify navigation suspended
        assert len(agent.agent_state.suspended_goals) == 1
        assert agent.agent_state.current_executor_type == 'battle'

        # Step 3: Battle completes
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )
        agent.step(test_game_state)

        # Verify battle completed and state cleared
        assert agent.agent_state.current_executor is None

        # Step 4: Resume navigation
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No current executor'
        )
        agent.step(test_game_state)

        # Verify navigation resumed
        assert len(agent.agent_state.suspended_goals) == 0
        assert agent.agent_state.current_executor_type == 'navigation'

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_empty_stack_warning(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        caplog
    ):
        """Test that warning is logged when trying to resume from empty stack."""
        agent = MVPHierarchicalAgent()
        agent.agent_state.suspended_goals = []

        # Try to resume (should log warning)
        with caplog.at_level('WARNING'):
            agent._resume_suspended_goal()

        assert 'Attempted to resume suspended goal but stack is empty' in caplog.text
        assert agent.agent_state.current_executor is None

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_state_preservation_across_suspend_resume(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls
    ):
        """Test that executor state is preserved across suspend/resume."""
        mock_nav_executor = MagicMock()
        mock_nav_executor.get_state.return_value = {
            'phase': 'navigating',
            'path': ['UP', 'UP', 'RIGHT'],
            'current_position': (10, 15)
        }
        mock_navigation_cls.return_value = mock_nav_executor

        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_nav_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Test'}

        # Suspend
        agent._suspend_current_goal()

        # Verify state captured
        suspended = agent.agent_state.suspended_goals[0]
        assert suspended['executor_state']['phase'] == 'navigating'
        assert suspended['executor_state']['path'] == ['UP', 'UP', 'RIGHT']
        assert suspended['executor_state']['current_position'] == (10, 15)

        # Resume
        agent._resume_suspended_goal()

        # Verify state restored
        mock_nav_executor.restore_state.assert_called_once_with({
            'phase': 'navigating',
            'path': ['UP', 'UP', 'RIGHT'],
            'current_position': (10, 15)
        })


# ============================================================================
# Test Class: Planner Integration
# ============================================================================

class TestPlannerIntegration:
    """Test planner integration."""

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_planner_called_when_no_executor(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_navigation
    ):
        """Test that planner is called when no current executor."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_navigation

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent with no executor
        agent = MVPHierarchicalAgent()

        # Execute step
        agent.step(test_game_state)

        # Verify planner was called
        mock_planner.create_plan.assert_called_once_with(
            perception=mock_perception_result,
            state_data=test_game_state,
            recent_logs=agent.agent_state.recent_logs
        )

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_planner_reasoning_captured_in_logs(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_battle
    ):
        """Test that planner reasoning is captured in recent_logs."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_battle

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='in_progress', summary=None
        )
        mock_battle_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()

        # Execute step
        agent.step(test_game_state)

        # Verify reasoning captured in logs
        assert len(agent.agent_state.recent_logs) > 0
        log_entry = agent.agent_state.recent_logs[-1]
        assert 'PLAN:' in log_entry
        assert 'Reasoning:' in log_entry
        assert mock_plan_result_battle.reasoning in log_entry

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_planner_creates_valid_plan_for_each_executor_type(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_navigation,
        mock_plan_result_battle,
        mock_plan_result_general
    ):
        """Test that planner can create plans for all executor types."""
        # Setup common mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value

        # Test navigation plan
        mock_planner.create_plan.return_value = mock_plan_result_navigation
        mock_nav_executor = MagicMock()
        mock_nav_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_nav_executor

        agent = MVPHierarchicalAgent()
        agent.step(test_game_state)
        assert agent.agent_state.current_executor_type == 'navigation'

        # Test battle plan
        agent.agent_state.current_executor = None
        agent.agent_state.current_executor_type = None
        mock_planner.create_plan.return_value = mock_plan_result_battle
        mock_battle_executor = MagicMock()
        mock_battle_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='in_progress', summary=None
        )
        mock_battle_cls.return_value = mock_battle_executor

        agent.step(test_game_state)
        assert agent.agent_state.current_executor_type == 'battle'

        # Test general plan
        agent.agent_state.current_executor = None
        agent.agent_state.current_executor_type = None
        mock_planner.create_plan.return_value = mock_plan_result_general
        mock_general_executor = MagicMock()
        mock_general_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='in_progress', summary=None
        )
        mock_general_cls.return_value = mock_general_executor

        agent.step(test_game_state)
        assert agent.agent_state.current_executor_type == 'general'


# ============================================================================
# Test Class: Memory Storage
# ============================================================================

class TestMemoryStorage:
    """Test memory storage."""

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_memory_stores_after_each_step(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test that memory.store is called after each step."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = 'Test'

        # Execute step
        agent.step(test_game_state)

        # Verify memory.store called
        agent.memory.store.assert_called_once()
        call_args = agent.memory.store.call_args
        assert call_args.kwargs['step_number'] == 1
        assert call_args.kwargs['raw_state'] == test_game_state
        assert call_args.kwargs['perception'] == mock_perception_result
        assert call_args.kwargs['actions'] == ['UP']

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_all_llm_outputs_aggregated(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_plan_result_navigation
    ):
        """Test that LLM outputs from all modules are aggregated correctly."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No current executor'
        )

        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = mock_plan_result_navigation

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()

        # Execute step (planner will be called)
        agent.step(test_game_state)

        # Verify aggregated LLM outputs
        call_args = agent.memory.store.call_args
        llm_outputs = call_args.kwargs['llm_outputs']

        # Should have perception outputs
        assert 'scene_description' in llm_outputs

        # Should have planner outputs
        assert 'planner_reasoning' in llm_outputs
        assert 'planner_executor_choice' in llm_outputs
        assert 'planner_goal' in llm_outputs

        assert llm_outputs['planner_reasoning'] == mock_plan_result_navigation.reasoning
        assert llm_outputs['planner_executor_choice'] == 'navigation'

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_step_count_increments(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test that step count increments properly."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = 'Test'

        assert agent.step_count == 0

        # Execute multiple steps
        agent.step(test_game_state)
        assert agent.step_count == 1

        agent.step(test_game_state)
        assert agent.step_count == 2

        agent.step(test_game_state)
        assert agent.step_count == 3


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases."""

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_missing_frame_in_game_state(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        mock_perception_result
    ):
        """Test handling of missing frame in game_state."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = 'Test'

        # Game state without frame
        game_state = {'player': {}, 'map': {}}

        # Should not crash
        result = agent.step(game_state)

        # Memory store should still be called (with None frame handled)
        agent.memory.store.assert_called_once()

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_executor_returns_empty_actions(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test handling when executor returns empty actions."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=[],  # Empty actions
            status='in_progress',
            summary=None
        )
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = 'Test'

        # Execute step
        result = agent.step(test_game_state)

        # Should return empty actions without crashing
        assert result['action'] == []

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_executor_fails_during_execution(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result,
        mock_executor_result_failed
    ):
        """Test handling when executor fails during execution."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True, reason='Valid'
        )

        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = mock_executor_result_failed
        mock_navigation_cls.return_value = mock_executor

        # Create agent
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {'target': None, 'description': 'Navigate'}

        # Execute step
        result = agent.step(test_game_state)

        # Executor should be cleared after failure
        assert agent.agent_state.current_executor is None
        assert agent.agent_state.current_executor_type is None

        # Failure should be logged
        assert len(agent.agent_state.recent_logs) > 0
        assert 'Failed to find path to target' in agent.agent_state.recent_logs[-1]

    @patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory')
    @patch('custom_agent.mvp_hierarchical.agent.LangChainVLM')
    @patch('custom_agent.mvp_hierarchical.agent.PerceptionModule')
    @patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector')
    @patch('custom_agent.mvp_hierarchical.agent.Planner')
    @patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.BattleExecutor')
    @patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor')
    def test_saliency_check_fails(
        self,
        mock_general_cls,
        mock_battle_cls,
        mock_navigation_cls,
        mock_planner_cls,
        mock_saliency_cls,
        mock_perception_cls,
        mock_vlm_cls,
        mock_memory_cls,
        test_game_state,
        mock_perception_result
    ):
        """Test handling when saliency check raises exception."""
        # Setup mocks
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = mock_perception_result

        mock_saliency = mock_saliency_cls.return_value
        # Saliency check raises exception
        mock_saliency.check_validity.side_effect = Exception("Saliency check failed")

        # Create agent
        agent = MVPHierarchicalAgent()

        # Should propagate exception (no error handling as per requirements)
        with pytest.raises(Exception, match="Saliency check failed"):
            agent.step(test_game_state)
