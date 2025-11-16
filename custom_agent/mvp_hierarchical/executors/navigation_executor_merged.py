"""
Navigation executor - handles navigation-only tasks (no interaction).

Heavily adapted from NavigationAgent's pathfinding logic.
Uses A* pathfinding to reach chosen targets (target selection done by planner).

TODO: refactor common logic with NavigationAgent into shared utils.
"""

from typing import List, Optional, Literal, Union, Tuple, TYPE_CHECKING
import logging

from custom_agent.mvp_hierarchical.executors.base_executor import BaseExecutor, ExecutorResult
from custom_utils.navigation_targets import NavigationTarget
from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid, add_turn_actions
from custom_utils.map_extractor import get_player_centered_grid

#NT: Dialogue Detection
from custom_utils.detectors import detect_battle, detect_dialogue
import numpy as np

if TYPE_CHECKING:
    from custom_agent.mvp_hierarchical.modules.perception import PerceptionResult

# Type alias for goal parameter
NavigationGoal = Union[NavigationTarget, dict]

logger = logging.getLogger(__name__)


class NavigationExecutor(BaseExecutor):
    """
    Navigation executor - executes paths to chosen targets.

    Phases:
    - idle: Start new navigation (expects target in goal)
    - navigating: Execute path in small batches

    Planner chooses target, this executor just executes the path.
    No interaction phase - just navigates and returns control to planner.
    """

    def __init__(
        self,
        action_batch_size: int = 100,
        stuck_threshold: int = 1,
        movement_mode: Literal["naive", "facing_aware"] = "naive"
    ):
        """
        Initialize NavigationExecutor.

        Args:
            action_batch_size: Number of actions to emit per step
            stuck_threshold: Steps without progress before failing
            movement_mode: Movement mode ("naive" or "facing_aware")
        """
        super().__init__()
        self.action_batch_size = action_batch_size
        self.stuck_threshold = stuck_threshold
        self.movement_mode = movement_mode
        self.pathfinder = AStarPathfinder()

        # Internal state (phase and navigation data)
        # Note: Using dict instead of TypedDict for simplicity and compatibility with base class
        self.internal_state: dict = {
            'phase': 'idle',
            'current_target': None,  # Optional[NavigationTarget]
            'current_path': [],  # List[str]
            'path_index': 0,  # int
            'steps_since_progress': 0,  # int
            'last_player_pos': None,  # Optional[Tuple[int, int]]
            'last_player_map': None,  # Optional[str]
            'facing_verified': False,  # bool
            'awaiting_probe_result': False,  # bool
            'naive_path': [],  # List[str]
        }

        # Current state from game (updated each step via _update_state)
        self.player_map_tile_x: Optional[int] = None
        self.player_map_tile_y: Optional[int] = None
        self.player_map: Optional[str] = None
        self.player_facing: Optional[str] = None
        self.traversability_map: Optional[List[List[str]]] = None

        logger.info(f"Initialized NavigationExecutor: batch_size={action_batch_size}, "
                   f"stuck_threshold={stuck_threshold}, movement_mode={movement_mode}")

    def execute_step(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: NavigationGoal
    ) -> ExecutorResult:
        """
        Execute navigation step (small batch).

        Goal format: Expected to contain NavigationTarget chosen by planner.
        Can be either:
        - NavigationTarget object directly
        - dict with 'target' key containing NavigationTarget

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: NavigationTarget or dict containing NavigationTarget with 'target' key

        Returns:
            ExecutorResult with actions and status
        """
        # Update current state from game
        self._update_state(state_data)

        if self.internal_state['phase'] == 'idle':
            return self._start_navigation(perception, state_data, goal)
        elif self.internal_state['phase'] == 'navigating':
            return self._execute_navigation(perception, state_data)
        else:
            logger.error(f"Unknown phase: {self.internal_state['phase']}")
            return ExecutorResult(
                actions=[],
                status='failed',
                summary=f"Unknown phase: {self.internal_state['phase']}"
            )

    def is_still_valid(
        self,
        state_data: dict,
        perception: 'PerceptionResult'
    ) -> bool:
        """
        Check if navigation is still valid.

        Invalid if:
        - Battle started (is_in_battle=True)
        - Not in overworld (game_state != 'overworld')

        Args:
            state_data: Game state data
            perception: Current perception result

        Returns:
            bool: True if still valid, False if need to replan
        """
        game_state = state_data.get('game', {})

        # Check if battle started
        if game_state.get('is_in_battle', False):
            logger.info("Navigation executor is invalid - battle started")
            return False
        
        #NT: Other checks
        frame = state_data.get('frame')
        if frame is not None:
            frame = np.array(frame)
            if detect_dialogue(frame):
                logger.info("Navigation executor is invalid - dialogue detected")
                return False
            if detect_battle(frame):
                logger.info("Navigation executor is invalid - battle detected")
                return False

        # Check if still in overworld
        # we remove this cos its not working
        # TODO: figure out handle dialog
        # if game_state.get('game_state') != 'overworld':
        #     logger.info(f"Navigation executor is invalid - not in overworld. Is in {game_state.get('game_state')}")
        #     return False

        return True

    def reset(self):
        """Reset navigation executor to initial idle state."""
        self.internal_state = {
            'phase': 'idle',
            'current_target': None,
            'current_path': [],
            'path_index': 0,
            'steps_since_progress': 0,
            'last_player_pos': None,
            'last_player_map': None,
            'facing_verified': False,
            'awaiting_probe_result': False,
            'naive_path': [],
        }

    def _update_state(self, state_data: dict) -> None:
        """
        Extract and update state from state_data (player + map).

        Adapted from NavigationAgent._update_state.

        Args:
            state_data: Game state data from emulator
        """
        player_data: dict = state_data.get('player', {})

        # Extract position (nested in position dict)
        position = player_data.get('position')
        if position is None or not isinstance(position, dict):
            logger.warning(f"Failed to extract player position. player_data keys: {list(player_data.keys())}")
            self.player_map_tile_x = None
            self.player_map_tile_y = None
        else:
            self.player_map_tile_x = position.get('x')
            self.player_map_tile_y = position.get('y')
            if self.player_map_tile_x is None or self.player_map_tile_y is None:
                logger.warning(f"Player position missing x or y. position: {position}")

        # Extract location (map)
        self.player_map = player_data.get('location')
        if self.player_map is None:
            logger.warning(f"Failed to extract player location. player_data keys: {list(player_data.keys())}")

        # Note: Facing direction is inferred during navigation via probe-and-process
        # (first action of path acts as a probe to determine facing)

        # Extract traversability map from visual_map
        map_data = state_data.get('map', {})
        self.traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)]
        )

    def _start_navigation(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: NavigationGoal
    ) -> ExecutorResult:
        """
        Start new navigation task.

        Adapted from NavigationAgent._start_new_navigation (step 4 onwards).

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: NavigationTarget or dict containing target

        Returns:
            ExecutorResult with first action(s)
        """
        logger.info("Starting new navigation phase")

        # Extract target from goal
        # Goal can be NavigationTarget directly or dict with 'target' key
        target: Optional[NavigationTarget] = None
        if isinstance(goal, NavigationTarget):
            target = goal
        elif isinstance(goal, dict) and 'target' in goal:
            target = goal['target']
        else:
            logger.error(f"Invalid goal format: {type(goal)}. Expected NavigationTarget or dict with 'target' key")
            return ExecutorResult(
                actions=[],
                status='failed',
                summary="Invalid goal format - no NavigationTarget provided"
            )

        # Extract people obstacles from perception
        people_obstacles = self._extract_people_obstacles(perception)
        if people_obstacles:
            logger.info(f"Detected {len(people_obstacles)} people as obstacles: {people_obstacles}")

        # Plan path to target
        logger.info(f"Planning path to target: {target.description}")
        path = self._plan_path_to_target(target, state_data, people_obstacles)

        if not path:
            logger.warning("No path found to target")
            return ExecutorResult(
                actions=[],
                status='failed',
                summary="No path found to target"
            )

        # Append exit step for doors (after planning, before execution)
        if target.type == "door" and target.exit_direction:
            exit_action = self._direction_to_action(target.exit_direction)
            path.append(exit_action)
            logger.info(f"Appended exit step '{exit_action}' for door")

        # Update navigation state
        player_map_tile_pos = (self.player_map_tile_x, self.player_map_tile_y)
        self.internal_state['phase'] = 'navigating'
        self.internal_state['current_target'] = target
        self.internal_state['path_index'] = 0
        self.internal_state['steps_since_progress'] = 0
        self.internal_state['last_player_pos'] = player_map_tile_pos
        self.internal_state['last_player_map'] = self.player_map

        # Setup probe-and-process for facing_aware mode
        if self.movement_mode == "facing_aware" and not self.internal_state['facing_verified']:
            self.internal_state['naive_path'] = path
            self.internal_state['awaiting_probe_result'] = True
            self.internal_state['current_path'] = []  # Empty until probe is processed
            logger.info(f"Facing-aware mode: emitting first action '{path[0]}' as probe")
            return ExecutorResult(actions=[path[0]], status='in_progress')
        else:
            # Normal mode or facing already verified
            self.internal_state['naive_path'] = []
            self.internal_state['awaiting_probe_result'] = False
            self.internal_state['current_path'] = path
            logger.info(f"Path planned with {len(path)} actions")
            return self._emit_next_actions()

    def _execute_navigation(
        self,
        perception: 'PerceptionResult',
        state_data: dict
    ) -> ExecutorResult:
        """
        Execute path step-by-step, check for arrival or stuck.

        Adapted from NavigationAgent._execute_navigation.

        Args:
            perception: Current perception result
            state_data: Game state data

        Returns:
            ExecutorResult with next action(s)
        """
        # Check if reached target
        if self._has_reached_target(state_data):
            logger.info("Reached target, completing navigation")
            self.internal_state['phase'] = 'idle'
            target_desc = self.internal_state['current_target'].description if self.internal_state['current_target'] else "target"
            return ExecutorResult(
                actions=['A'],
                status='completed',
                summary=f"Reached target: {target_desc}"
            )

        # Track player state (map + position) to detect progress
        current_pos = (self.player_map_tile_x, self.player_map_tile_y)
        current_map = self.player_map

        position_changed = (current_map != self.internal_state['last_player_map'] or
                          current_pos != self.internal_state['last_player_pos'])

        # Check if awaiting probe result
        if self.internal_state['awaiting_probe_result']:
            self._update_probe_result(position_changed)

        # Check if state changed (progress made)
        if position_changed:
            # Progress detected (map changed OR position changed)
            if current_map != self.internal_state['last_player_map']:
                logger.info(f"Progress: Map changed from '{self.internal_state['last_player_map']}' to '{current_map}'")
            else:
                logger.debug(f"Progress: Position changed from {self.internal_state['last_player_pos']} to {current_pos}")

            self.internal_state['steps_since_progress'] = 0
            self.internal_state['last_player_pos'] = current_pos
            self.internal_state['last_player_map'] = current_map
        else:
            # No progress (map AND position unchanged)
            self.internal_state['steps_since_progress'] += 1
            logger.warning(f"No progress: {self.internal_state['steps_since_progress']}/{self.stuck_threshold}")

        # Check if stuck (no progress for threshold steps)
        if self.internal_state['steps_since_progress'] >= self.stuck_threshold:
            logger.warning(f"Stuck for {self.internal_state['steps_since_progress']} steps (threshold={self.stuck_threshold})")
            self.internal_state['phase'] = 'idle'
            return ExecutorResult(
                actions=['A'],
                status='failed',
                summary=f"Stuck - no progress for {self.stuck_threshold} steps"
            )

        # Check if path is exhausted
        if self.internal_state['path_index'] >= len(self.internal_state['current_path']):
            logger.warning("Path exhausted but target not reached")
            self.internal_state['phase'] = 'idle'
            return ExecutorResult(
                actions=['A'],
                status='failed',
                summary="Path exhausted without reaching target"
            )

        # Emit next batch of actions
        return self._emit_next_actions()

    def _emit_next_actions(self) -> ExecutorResult:
        """
        Emit next batch of actions from current path.

        Returns:
            ExecutorResult with next actions (up to batch_size)
        """
        if not self.internal_state['current_path']:
            return ExecutorResult(actions=[], status='in_progress')

        actions_to_emit = []
        for _ in range(self.action_batch_size):
            if self.internal_state['path_index'] >= len(self.internal_state['current_path']):
                break
            actions_to_emit.append(self.internal_state['current_path'][self.internal_state['path_index']])
            self.internal_state['path_index'] += 1

        if actions_to_emit:
            logger.debug(
                f"Emitting actions [{self.internal_state['path_index'] - len(actions_to_emit)}:"
                f"{self.internal_state['path_index']}]/{len(self.internal_state['current_path'])}: {actions_to_emit}"
            )

        return ExecutorResult(actions=actions_to_emit, status='in_progress')

    def _update_probe_result(self, position_changed: bool) -> None:
        """
        Process probe result to determine facing direction.

        Adapted from NavigationAgent._update_probe_result.

        Args:
            position_changed: Whether player position changed after probe action
        """
        logger.info("Processing probe result to determine facing direction")

        # Check if player moved (position changed)
        probe_action = self.internal_state['naive_path'][0]

        # Map probe action to direction
        ACTION_TO_DIRECTION = {
            "UP": "North",
            "DOWN": "South",
            "LEFT": "West",
            "RIGHT": "East"
        }
        inferred_facing = ACTION_TO_DIRECTION.get(probe_action, "North")

        if position_changed:
            # Player moved - was already facing that direction
            logger.info(f"Probe result: Position changed. Player was already facing {inferred_facing}")
            # First action consumed, process remaining path
            remaining_naive_path = self.internal_state['naive_path'][1:]
        else:
            # Player just turned - was NOT facing that direction
            logger.info(f"Probe result: Position unchanged. Player turned to face {inferred_facing}")
            # First action was turn, not consumed from naive path
            remaining_naive_path = self.internal_state['naive_path']

        # Update player facing
        self.player_facing = inferred_facing
        self.internal_state['facing_verified'] = True

        # Process remaining naive path with verified facing
        processed_path = add_turn_actions(remaining_naive_path, inferred_facing)

        # Update path and clear probe state
        self.internal_state['current_path'] = processed_path
        self.internal_state['path_index'] = 0
        self.internal_state['awaiting_probe_result'] = False

        logger.info(f"Processed remaining path: {len(processed_path)} actions")

    def _direction_to_action(self, direction: Tuple[int, int]) -> str:
        """
        Convert direction tuple (dx, dy) to action string.

        Args:
            direction: Direction tuple (dx, dy)

        Returns:
            Action string ("UP", "DOWN", "LEFT", "RIGHT")
        """
        dx, dy = direction
        direction_map = {
            (0, -1): "UP",
            (0, 1): "DOWN",
            (-1, 0): "LEFT",
            (1, 0): "RIGHT"
        }
        action = direction_map.get((dx, dy))
        if not action:
            logger.error(f"Invalid direction tuple: {direction}")
            return "DOWN"  # Default fallback
        return action

    def _plan_path_to_target(
        self,
        target: NavigationTarget,
        state_data: dict,
        obstacles: Optional[List[Tuple[int, int]]] = None
    ) -> List[str]:
        """
        Use A* pathfinder to plan path to target.

        Adapted from NavigationAgent._plan_path_to_target.

        Args:
            target: Navigation target
            state_data: Game state data
            obstacles: Optional list of (x, y) obstacle positions to avoid

        Returns:
            List of actions to reach target
        """
        # Player is always at local tile (7, 7)
        start_local_tile = (7, 7)
        goal_local_tile = target.local_tile_position

        # Get player facing direction
        start_facing = self.player_facing if self.player_facing else 'North'

        # Use traversability map from state
        terrain = TerrainGrid(self.traversability_map)

        # For facing_aware mode without verified facing, get naive path first
        # The first action will act as a probe to determine actual facing
        if self.movement_mode == "facing_aware" and not self.internal_state['facing_verified']:
            movement_mode = "naive"
        else:
            movement_mode = self.movement_mode

        path = self.pathfinder.find_path(
            start_local_tile=start_local_tile,
            goal_local_tile=goal_local_tile,
            start_facing=start_facing,
            terrain=terrain,
            obstacles=obstacles,
            movement_mode=movement_mode
        )

        logger.info(f"Planned path to {target.description}: {len(path)} actions (mode={movement_mode}, obstacles={len(obstacles) if obstacles else 0})")

        return path

    def _extract_people_obstacles(self, perception: 'PerceptionResult') -> List[Tuple[int, int]]:
        """
        Extract people obstacles from navigation targets.

        Args:
            perception: PerceptionResult containing navigation targets

        Returns:
            List of (x, y) local tile positions for detected people
        """
        people_obstacles = []

        for target in perception.navigation_targets:
            # Check if this target has a detected_object with source "people_detection"
            if (target.detected_object is not None and
                target.detected_object.source == "people_detection"):
                if target.local_tile_position != (7, 7):
                    people_obstacles.append(target.local_tile_position)
                    logger.debug(f"Added person obstacle at {target.local_tile_position}: {target.description}")

        return people_obstacles

    def _has_reached_target(self, state_data: dict) -> bool:
        """
        Check if player reached target (handles map transitions and distance).

        Adapted from NavigationAgent._has_reached_target.
        TODO: Refactor common logic with navigation_agent.py

        Args:
            state_data: Game state data

        Returns:
            True if target reached based on type-specific logic
        """
        current_target = self.internal_state['current_target']
        if not current_target:
            return False

        current_map = self.player_map
        target_source_map = current_target.source_map_location
        player_pos = (self.player_map_tile_x, self.player_map_tile_y)
        target_pos = current_target.map_tile_position

        # Check if map changed (indicates transition occurred)
        map_changed = (target_source_map is not None and
                      current_map is not None and
                      current_map != target_source_map)

        # Calculate distance on current map
        distance = abs(player_pos[0] - target_pos[0]) + abs(player_pos[1] - target_pos[1])

        target_type = current_target.type
        target_description = current_target.description

        if target_type in ["door", "stairs"]:
            # Both doors and stairs: reached when map transitions
            # Path handles difference: stairs ends ON tile, doors end BEYOND (via exit step)
            if map_changed:
                logger.info(f"{target_type.capitalize()} target '{target_description}' reached - map transitioned from '{target_source_map}' to '{current_map}'")
            return map_changed

        elif target_type == "boundary":
            # Boundaries: reached when map changes OR distance is 0
            if map_changed:
                logger.info(f"Boundary target '{target_description}' reached - map transitioned from '{target_source_map}' to '{current_map}'")
            elif distance == 0:
                logger.info(f"Boundary target '{target_description}' reached - player on boundary tile")
            return map_changed or distance == 0

        elif target_type == "object":
            # Objects (NPCs, items): distance-based check on same map
            # Adjacent or on top (distance <= 1)
            reached = distance <= 1
            if reached:
                logger.info(f"Object target '{target_description}' reached at distance {distance}")
            return reached

        else:
            logger.warning(f"Unknown target type: {target_type}")
            return False
