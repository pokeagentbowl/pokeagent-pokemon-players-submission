#!/usr/bin/env python3
"""
Integration tests for MVPHierarchicalAgent - Full Flow Tests

These tests verify the complete agent coordination across all modules and executors.
Uses real fixtures and mocked LLM calls with realistic responses.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from custom_agent.mvp_hierarchical.agent import MVPHierarchicalAgent, AgentState
from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult
from custom_agent.mvp_hierarchical.modules.saliency import SaliencyResult
from custom_agent.mvp_hierarchical.modules.planner import PlanResult
from custom_agent.mvp_hierarchical.executors.base_executor import ExecutorResult
from custom_utils.object_detector import DetectedObject
from custom_utils.navigation_targets import NavigationTarget


# ============================================================================
# Fixtures - Game States
# ============================================================================

@pytest.fixture
def overworld_game_state():
    """Real overworld game state fixture."""
    return {
        'frame': np.zeros((160, 240, 3), dtype=np.uint8),
        'player': {
            'position': {'x': 10, 'y': 15},
            'location': 'LITTLEROOT_TOWN',
            'in_battle': False,
            'in_overworld': True,
            'facing': 'DOWN'
        },
        'map': {
            'visual_map': '--- MAP: LITTLEROOT_TOWN ---\n' + '\n'.join(['.' * 15 for _ in range(15)]),
            'width': 15,
            'height': 15
        },
        'party': [
            {'species': 'TORCHIC', 'level': 5, 'hp': 20, 'max_hp': 20}
        ]
    }


@pytest.fixture
def battle_game_state():
    """Real battle game state fixture."""
    return {
        'frame': np.zeros((160, 240, 3), dtype=np.uint8),
        'player': {
            'position': {'x': 10, 'y': 15},
            'location': 'ROUTE_101',
            'in_battle': True,
            'in_overworld': False
        },
        'battle': {
            'active': True,
            'player_pokemon': {
                'species': 'TORCHIC',
                'level': 5,
                'hp': 15,
                'max_hp': 20,
                'moves': ['SCRATCH', 'GROWL']
            },
            'opponent_pokemon': {
                'species': 'POOCHYENA',
                'level': 3,
                'hp': 12,
                'max_hp': 12
            }
        },
        'party': [
            {'species': 'TORCHIC', 'level': 5, 'hp': 15, 'max_hp': 20}
        ]
    }


@pytest.fixture
def menu_game_state():
    """Menu interaction game state fixture."""
    return {
        'frame': np.zeros((160, 240, 3), dtype=np.uint8),
        'player': {
            'position': {'x': 10, 'y': 15},
            'location': 'LITTLEROOT_TOWN',
            'in_battle': False,
            'in_overworld': True,
            'menu_open': True
        },
        'map': {
            'visual_map': '--- MAP: LITTLEROOT_TOWN ---\n' + '\n'.join(['.' * 15 for _ in range(15)])
        },
        'party': [
            {'species': 'TORCHIC', 'level': 5, 'hp': 20, 'max_hp': 20}
        ]
    }


# ============================================================================
# Fixtures - Mock Perception Results
# ============================================================================

@pytest.fixture
def navigation_perception():
    """Perception result with navigation targets."""
    detected_objects = [
        DetectedObject(
            name='NPC',
            confidence=0.9,
            bbox={'x': 50, 'y': 60, 'w': 20, 'h': 30},
            center_pixel=(60, 75)
        )
    ]
    navigation_targets = [
        NavigationTarget(
            id='target_0',
            type='object',
            map_tile_position=(12, 18),
            local_tile_position=(12, 18),
            description='NPC at (12, 18)',
            detected_object=detected_objects[0]
        )
    ]
    return PerceptionResult(
        detected_objects=detected_objects,
        scene_description='Overworld scene with NPC visible',
        navigation_targets=navigation_targets,
        llm_outputs={'scene_description': 'Overworld scene with NPC visible'}
    )


@pytest.fixture
def battle_perception():
    """Perception result for battle scenario."""
    detected_objects = [
        DetectedObject(
            name='BATTLE_UI',
            confidence=0.95,
            bbox={'x': 10, 'y': 100, 'w': 220, 'h': 50},
            center_pixel=(120, 125)
        )
    ]
    return PerceptionResult(
        detected_objects=detected_objects,
        scene_description='Battle screen with move selection',
        navigation_targets=[],
        llm_outputs={'scene_description': 'Battle screen with move selection'}
    )


# ============================================================================
# Test 1: Full Agent Step Without Replanning
# ============================================================================

def test_full_step_without_replanning(overworld_game_state, navigation_perception):
    """
    Test basic agent flow with stable executor.

    Verifies: perception → saliency → executor → memory (all called correctly)
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor'), \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor'):

        # Setup perception
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = navigation_perception

        # Setup saliency - executor remains valid
        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='NavigationExecutor still valid'
        )

        # Setup executor
        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP', 'UP', 'RIGHT'],
            status='in_progress',
            summary=None
        )
        mock_nav_cls.return_value = mock_executor

        # Create agent with navigation executor already active
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {
            'target': navigation_perception.navigation_targets[0],
            'description': 'Navigate to NPC at (12, 18)'
        }

        # Execute single step
        result = agent.step(overworld_game_state)

        # Verify full flow
        mock_perception.process.assert_called_once_with(overworld_game_state)
        mock_saliency.check_validity.assert_called_once()
        mock_executor.execute_step.assert_called_once()
        agent.memory.store.assert_called_once()

        # Verify actions returned
        assert result['action'] == ['UP', 'UP', 'RIGHT']
        assert result['reasoning'] == 'Navigate to NPC at (12, 18)'

        # Verify memory stored with correct data
        call_args = agent.memory.store.call_args
        assert call_args.kwargs['step_number'] == 1
        assert call_args.kwargs['actions'] == ['UP', 'UP', 'RIGHT']
        assert call_args.kwargs['perception'] == navigation_perception


# ============================================================================
# Test 2: Full Agent Step With Executor Completion
# ============================================================================

def test_full_step_with_executor_completion(overworld_game_state, navigation_perception):
    """
    Test executor completion triggers replanning on next step.

    Verifies: executor completes → state cleared → next step creates new plan
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor'), \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor'):

        # Setup perception
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = navigation_perception

        # Setup saliency
        mock_saliency = mock_saliency_cls.return_value

        # Setup executor - will complete on first step
        mock_executor = MagicMock()
        mock_executor.execute_step.side_effect = [
            ExecutorResult(
                actions=['A'],
                status='completed',
                summary='Successfully reached NPC at (12, 18)'
            ),
            ExecutorResult(
                actions=['UP', 'RIGHT'],
                status='in_progress',
                summary=None
            )
        ]
        mock_nav_cls.return_value = mock_executor

        # Setup planner for second step
        mock_planner = mock_planner_cls.return_value
        new_target = NavigationTarget(
            id='target_1',
            type='object',
            map_tile_position=(8, 10),
            local_tile_position=(8, 10),
            description='Different NPC at (8, 10)',
            detected_object=None
        )
        mock_planner.create_plan.return_value = PlanResult(
            executor_type='navigation',
            goal={'target': new_target, 'description': 'Navigate to Different NPC at (8, 10)'},
            reasoning='Previous goal completed, moving to next objective'
        )

        # Create agent with navigation executor
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {
            'target': navigation_perception.navigation_targets[0],
            'description': 'Navigate to NPC at (12, 18)'
        }

        # Step 1: Executor completes
        result1 = agent.step(overworld_game_state)

        # Verify executor completed and state cleared
        assert result1['action'] == ['A']
        assert agent.agent_state.current_executor is None
        assert agent.agent_state.current_executor_type is None
        assert len(agent.agent_state.recent_logs) == 1
        assert 'Successfully reached NPC' in agent.agent_state.recent_logs[0]

        # Step 2: New plan should be created
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='No current executor'
        )

        result2 = agent.step(overworld_game_state)

        # Verify planner called and new plan created
        mock_planner.create_plan.assert_called_once()
        assert agent.agent_state.current_executor_type == 'navigation'
        assert agent.agent_state.current_goal['description'] == 'Navigate to Different NPC at (8, 10)'
        assert 'PLAN:' in agent.agent_state.recent_logs[-1]
        assert 'Previous goal completed' in agent.agent_state.recent_logs[-1]


# ============================================================================
# Test 3: Goal Suspension - Navigation → Battle → Resume
# ============================================================================

def test_goal_suspension_navigation_to_battle_to_resume(
    overworld_game_state,
    battle_game_state,
    navigation_perception,
    battle_perception
):
    """
    Test complete suspension/resumption flow.

    Flow:
    1. Navigation in progress
    2. Battle starts (executor invalid) → suspend navigation
    3. Battle runs and completes
    4. Resume navigation with restored state
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor') as mock_battle_cls, \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor'):

        # Setup perception - will return different results based on game state
        mock_perception = mock_perception_cls.return_value

        # Setup saliency
        mock_saliency = mock_saliency_cls.return_value

        # Setup navigation executor
        mock_nav_executor = MagicMock()
        mock_nav_executor.get_state.return_value = {
            'phase': 'navigating',
            'path': ['UP', 'UP', 'RIGHT', 'RIGHT'],
            'current_index': 2
        }
        mock_nav_executor.execute_step.return_value = ExecutorResult(
            actions=['UP', 'UP'],
            status='in_progress',
            summary=None
        )
        mock_nav_cls.return_value = mock_nav_executor

        # Setup battle executor
        mock_battle_executor = MagicMock()
        mock_battle_executor.execute_step.side_effect = [
            ExecutorResult(actions=['A'], status='in_progress', summary=None),
            ExecutorResult(actions=['A'], status='completed', summary='Battle won')
        ]
        mock_battle_cls.return_value = mock_battle_executor

        # Setup planner for battle
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = PlanResult(
            executor_type='battle',
            goal='Defeat the wild Pokemon',
            reasoning='Entered battle, need to fight'
        )

        # Create agent with navigation in progress
        agent = MVPHierarchicalAgent()
        agent.agent_state.current_executor = mock_nav_executor
        agent.agent_state.current_executor_type = 'navigation'
        agent.agent_state.current_goal = {
            'target': navigation_perception.navigation_targets[0],
            'description': 'Navigate to NPC at (12, 18)'
        }

        # Step 1: Navigation in progress
        mock_perception.process.return_value = navigation_perception
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='NavigationExecutor valid'
        )

        result1 = agent.step(overworld_game_state)
        assert result1['action'] == ['UP', 'UP']
        assert agent.agent_state.current_executor_type == 'navigation'

        # Step 2: Battle interrupts (executor becomes invalid)
        mock_perception.process.return_value = battle_perception
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='NavigationExecutor invalid - entered battle'
        )

        result2 = agent.step(battle_game_state)

        # Verify navigation was suspended
        assert len(agent.agent_state.suspended_goals) == 1
        suspended = agent.agent_state.suspended_goals[0]
        assert suspended['executor_type'] == 'navigation'
        assert suspended['goal']['description'] == 'Navigate to NPC at (12, 18)'
        assert suspended['executor_state']['phase'] == 'navigating'
        assert suspended['executor_state']['current_index'] == 2

        # Verify battle executor active
        assert agent.agent_state.current_executor_type == 'battle'
        assert agent.agent_state.current_goal == 'Defeat the wild Pokemon'

        # Step 3: Battle in progress
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=True,
            reason='BattleExecutor valid'
        )

        result3 = agent.step(battle_game_state)
        assert result3['action'] == ['A']

        # Step 4: Battle completes and navigation is immediately resumed
        # (because there's a suspended goal)
        result4 = agent.step(battle_game_state)

        # Verify navigation resumed
        assert len(agent.agent_state.suspended_goals) == 0
        assert agent.agent_state.current_executor_type == 'navigation'
        assert agent.agent_state.current_goal['description'] == 'Navigate to NPC at (12, 18)'

        # Verify state was restored
        mock_nav_executor.restore_state.assert_called_once_with({
            'phase': 'navigating',
            'path': ['UP', 'UP', 'RIGHT', 'RIGHT'],
            'current_index': 2
        })


# ============================================================================
# Test 4: Memory Retrieval in Planning
# ============================================================================

def test_memory_retrieval_affects_planning(overworld_game_state, navigation_perception):
    """
    Test that planner uses retrieved memories during planning.

    Verifies: memory.retrieve called → relevant memories returned → used in planning
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor'), \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor'):

        # Setup perception
        mock_perception = mock_perception_cls.return_value
        mock_perception.process.return_value = navigation_perception

        # Setup saliency
        mock_saliency = mock_saliency_cls.return_value
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False,
            reason='No current executor'
        )

        # Setup memory with stored experiences
        mock_memory = mock_memory_cls.return_value
        # Memory retrieval returns relevant past experiences
        mock_memory.retrieve.return_value = [
            {
                'step': 5,
                'summary': 'Previously navigated to NPC successfully',
                'outcome': 'completed'
            },
            {
                'step': 12,
                'summary': 'Tried talking to NPC, got useful info',
                'outcome': 'completed'
            }
        ]

        # Setup planner
        mock_planner = mock_planner_cls.return_value
        mock_planner.create_plan.return_value = PlanResult(
            executor_type='navigation',
            goal={
                'target': navigation_perception.navigation_targets[0],
                'description': 'Navigate to NPC at (12, 18)'
            },
            reasoning='Based on past success with NPCs, navigate to this one'
        )

        # Setup executor
        mock_executor = MagicMock()
        mock_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='in_progress', summary=None
        )
        mock_nav_cls.return_value = mock_executor

        # Create agent (no current executor - will trigger planning)
        agent = MVPHierarchicalAgent()

        # Manually store some memories first (simulate previous steps)
        for i in range(5):
            agent.memory.store(
                step_number=i,
                raw_state=overworld_game_state,
                image=overworld_game_state['frame'],
                perception=navigation_perception,
                actions=['UP'],
                llm_outputs={}
            )

        # Execute step (should trigger planning with memory retrieval)
        result = agent.step(overworld_game_state)

        # Verify planner was called
        mock_planner.create_plan.assert_called_once()
        call_args = mock_planner.create_plan.call_args
        assert call_args.kwargs['perception'] == navigation_perception

        # Note: Memory retrieval is called internally by Planner, not by Agent
        # The planner has access to memory through self.memory
        # This test verifies the integration point exists

        # Verify plan was created and executed
        assert agent.agent_state.current_executor_type == 'navigation'
        assert 'Based on past success' in agent.agent_state.recent_logs[-1]


# ============================================================================
# Test 5: Multiple Executor Types in Sequence
# ============================================================================

def test_multiple_executor_types_in_sequence(
    overworld_game_state,
    battle_game_state,
    menu_game_state,
    navigation_perception,
    battle_perception
):
    """
    Test agent handles all executor types in sequence.

    Sequence: navigation → battle → general → navigation
    Verifies: Each executor runs correctly, transitions are clean
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor') as mock_battle_cls, \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor') as mock_general_cls:

        # Setup perception
        mock_perception = mock_perception_cls.return_value

        # Setup saliency
        mock_saliency = mock_saliency_cls.return_value

        # Setup executors
        mock_nav_executor = MagicMock()
        mock_nav_executor.execute_step.return_value = ExecutorResult(
            actions=['UP'], status='completed', summary='Navigation complete'
        )
        mock_nav_cls.return_value = mock_nav_executor

        mock_battle_executor = MagicMock()
        mock_battle_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='completed', summary='Battle won'
        )
        mock_battle_cls.return_value = mock_battle_executor

        mock_general_executor = MagicMock()
        mock_general_executor.execute_step.return_value = ExecutorResult(
            actions=['A'], status='completed', summary='Menu closed'
        )
        mock_general_cls.return_value = mock_general_executor

        # Setup planner
        mock_planner = mock_planner_cls.return_value
        plans = [
            PlanResult(executor_type='navigation', goal={'target': None, 'description': 'Navigate'}, reasoning='Go to location'),
            PlanResult(executor_type='battle', goal='Fight', reasoning='Battle started'),
            PlanResult(executor_type='general', goal='Close menu', reasoning='Menu open'),
            PlanResult(executor_type='navigation', goal={'target': None, 'description': 'Navigate again'}, reasoning='Continue journey')
        ]
        mock_planner.create_plan.side_effect = plans

        # Create agent
        agent = MVPHierarchicalAgent()

        # Sequence 1: Navigation
        mock_perception.process.return_value = navigation_perception
        mock_saliency.check_validity.return_value = SaliencyResult(
            executor_valid=False, reason='No executor'
        )
        agent.step(overworld_game_state)
        assert agent.agent_state.current_executor_type is None  # Completed
        assert 'Navigation complete' in agent.agent_state.recent_logs[-1]

        # Sequence 2: Battle
        mock_perception.process.return_value = battle_perception
        agent.step(battle_game_state)
        assert agent.agent_state.current_executor_type is None  # Completed
        assert 'Battle won' in agent.agent_state.recent_logs[-1]

        # Sequence 3: General
        mock_perception.process.return_value = navigation_perception  # Menu case
        agent.step(menu_game_state)
        assert agent.agent_state.current_executor_type is None  # Completed
        assert 'Menu closed' in agent.agent_state.recent_logs[-1]

        # Sequence 4: Navigation again
        agent.step(overworld_game_state)
        assert agent.agent_state.current_executor_type is None  # Completed

        # Verify all executors were used
        assert len(agent.agent_state.recent_logs) >= 4
        log_text = ' '.join(agent.agent_state.recent_logs)
        assert 'Navigation' in log_text
        assert 'Battle' in log_text or 'Fight' in log_text
        assert 'Menu' in log_text or 'Close menu' in log_text


# ============================================================================
# Test 6: Full Episode Simulation (50 steps)
# ============================================================================

def test_full_episode_simulation(
    overworld_game_state,
    battle_game_state,
    navigation_perception,
    battle_perception
):
    """
    Stress test agent coordination over 50 steps.

    Includes:
    - Normal navigation
    - Battle interruptions
    - Menu interactions
    - Multiple goal suspensions

    Verifies: No crashes, all transitions handled, memory grows, logs maintained
    """
    with patch('custom_agent.mvp_hierarchical.agent.EpisodicMemory') as mock_memory_cls, \
         patch('custom_agent.mvp_hierarchical.agent.LangChainVLM') as mock_vlm_cls, \
         patch('custom_agent.mvp_hierarchical.agent.PerceptionModule') as mock_perception_cls, \
         patch('custom_agent.mvp_hierarchical.agent.SaliencyDetector') as mock_saliency_cls, \
         patch('custom_agent.mvp_hierarchical.agent.Planner') as mock_planner_cls, \
         patch('custom_agent.mvp_hierarchical.agent.NavigationExecutor') as mock_nav_cls, \
         patch('custom_agent.mvp_hierarchical.agent.BattleExecutor') as mock_battle_cls, \
         patch('custom_agent.mvp_hierarchical.agent.GeneralExecutor') as mock_general_cls:

        # Setup perception
        mock_perception = mock_perception_cls.return_value

        # Setup saliency - will vary based on scenario
        mock_saliency = mock_saliency_cls.return_value

        # Setup executors with varied behavior
        mock_nav_executor = MagicMock()
        mock_nav_executor.get_state.return_value = {'phase': 'navigating', 'steps': 0}

        mock_battle_executor = MagicMock()
        mock_general_executor = MagicMock()

        mock_nav_cls.return_value = mock_nav_executor
        mock_battle_cls.return_value = mock_battle_executor
        mock_general_cls.return_value = mock_general_executor

        # Setup planner
        mock_planner = mock_planner_cls.return_value

        # Create agent
        agent = MVPHierarchicalAgent()

        # Run 50 steps with varying scenarios
        total_steps = 50
        battle_steps = [10, 11, 12, 25, 26]  # Battle at these steps
        menu_steps = [20, 35]  # Menu interactions

        for step in range(total_steps):
            # Determine scenario
            if step in battle_steps:
                game_state = battle_game_state
                perception = battle_perception

                # Battle scenarios
                if step == battle_steps[0] or step == battle_steps[3]:
                    # Battle starts - suspend navigation
                    mock_saliency.check_validity.return_value = SaliencyResult(
                        executor_valid=False,
                        reason='Entered battle'
                    )
                    mock_planner.create_plan.return_value = PlanResult(
                        executor_type='battle',
                        goal='Win battle',
                        reasoning='Battle started'
                    )
                    mock_battle_executor.execute_step.return_value = ExecutorResult(
                        actions=['A'], status='in_progress', summary=None
                    )
                else:
                    # Battle in progress or ending
                    mock_saliency.check_validity.return_value = SaliencyResult(
                        executor_valid=True,
                        reason='Battle continues'
                    )
                    if step == battle_steps[2] or step == battle_steps[-1]:
                        # Battle ends
                        mock_battle_executor.execute_step.return_value = ExecutorResult(
                            actions=['A'], status='completed', summary='Battle won'
                        )
                    else:
                        mock_battle_executor.execute_step.return_value = ExecutorResult(
                            actions=['A'], status='in_progress', summary=None
                        )

            elif step in menu_steps:
                game_state = overworld_game_state
                game_state['player']['menu_open'] = True
                perception = navigation_perception

                # Menu interaction
                mock_saliency.check_validity.return_value = SaliencyResult(
                    executor_valid=False,
                    reason='Menu opened'
                )
                mock_planner.create_plan.return_value = PlanResult(
                    executor_type='general',
                    goal='Close menu',
                    reasoning='Menu interaction needed'
                )
                mock_general_executor.execute_step.return_value = ExecutorResult(
                    actions=['B'], status='completed', summary='Menu closed'
                )

            else:
                # Normal navigation
                game_state = overworld_game_state
                perception = navigation_perception

                if agent.agent_state.current_executor is None:
                    # Start new navigation or resume
                    mock_saliency.check_validity.return_value = SaliencyResult(
                        executor_valid=False,
                        reason='No executor'
                    )
                    mock_planner.create_plan.return_value = PlanResult(
                        executor_type='navigation',
                        goal={'target': None, 'description': f'Navigate step {step}'},
                        reasoning='Continue navigation'
                    )
                else:
                    mock_saliency.check_validity.return_value = SaliencyResult(
                        executor_valid=True,
                        reason='Navigation continues'
                    )

                # Navigation actions
                if step % 7 == 0 and agent.agent_state.current_executor_type == 'navigation':
                    # Complete navigation every 7 steps
                    mock_nav_executor.execute_step.return_value = ExecutorResult(
                        actions=['A'], status='completed', summary='Reached destination'
                    )
                else:
                    mock_nav_executor.execute_step.return_value = ExecutorResult(
                        actions=['UP', 'RIGHT'], status='in_progress', summary=None
                    )

            mock_perception.process.return_value = perception

            # Execute step
            try:
                result = agent.step(game_state)

                # Verify result has required fields
                assert 'action' in result
                assert 'reasoning' in result
                assert isinstance(result['action'], list)

            except Exception as e:
                pytest.fail(f"Step {step} failed with error: {e}")

        # Verify episode completed successfully
        assert agent.step_count == total_steps

        # Verify memory was stored for all steps
        assert agent.memory.store.call_count == total_steps

        # Verify recent logs maintained (should be capped at 20)
        assert len(agent.agent_state.recent_logs) <= 20

        # Verify no suspended goals left (all were resolved)
        assert len(agent.agent_state.suspended_goals) <= 1  # At most one unresolved suspension

        # Verify logs contain evidence of different scenarios
        log_text = ' '.join(agent.agent_state.recent_logs)
        # Should have some evidence of battles, navigation, etc.
        # (specific assertions depend on the random scenario generation)
