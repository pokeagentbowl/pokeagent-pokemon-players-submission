"""
Navigation executor NT - handles navigation-only tasks with advanced features.

Incorporates active tile caching, stuck detection, bandit fallback, and traversability patching.

Heavily adapted from NavigationAgentNT's advanced navigation logic.
Uses A* pathfinding to reach chosen targets (target selection done by planner).

TODO: refactor common logic with NavigationAgent into shared utils.
"""

from typing import List, Optional, Literal, Union, Tuple, TYPE_CHECKING, Dict
import logging
import numpy as np

from custom_agent.mvp_hierarchical.executors.base_executor import BaseExecutor, ExecutorResult
from custom_utils.navigation_targets_nt import NavigationTarget, generate_navigation_targets
from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid, add_turn_actions
from custom_utils.map_extractor_nt import get_player_centered_grid, extract_portal_connections
from custom_utils.log_to_active import (
    ensure_cache_directories, load_active_tile_index, save_active_tile_index,
    log_interaction, save_tile_to_cache, mark_and_save_tile,
    load_target_selection_counts, save_target_selection_counts,
    load_portal_connections, update_portal_connections_cache,
    INTERACTION_LOG_FILE
)

if TYPE_CHECKING:
    from custom_agent.mvp_hierarchical.modules.perception_nt import PerceptionResult

# Type alias for goal parameter
NavigationGoal = Union[NavigationTarget, dict]

logger = logging.getLogger(__name__)


class NavigationExecutorNT(BaseExecutor):
    """
    Navigation executor NT - executes paths to chosen targets with advanced features.

    Phases:
    - idle: Start new navigation (expects target in goal)
    - navigating: Execute path in small batches
    - interacting: Handle stuck situations or target interactions

    Planner chooses target, this executor just executes the path.
    Includes active tile caching, stuck detection, and bandit fallback.
    """

    def __init__(
        self,
        action_batch_size: int = 100,
        stuck_threshold: int = 3,
        movement_mode: Literal["naive", "facing_aware"] = "naive"
    ):
        """
        Initialize NavigationExecutorNT.

        Args:
            action_batch_size: Number of actions to emit per step
            stuck_threshold: Steps without progress before replanning
            movement_mode: Movement mode ("naive" or "facing_aware")
        """
        super().__init__()
        self.action_batch_size = action_batch_size
        self.stuck_threshold = stuck_threshold
        self.movement_mode = movement_mode
        self.pathfinder = AStarPathfinder()

        # Internal state (phase and navigation data)
        # Extended with NT features
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

            # NT features
            'failed_target_attempts': 0,  # int
            'initial_distance_to_target': None,  # Optional[float]
            'displacement_history': [],  # List[float]
            'bandit_mode_active': False,  # bool
            'bandit_iterations': 0,  # int
            'bandit_start_pos': None,  # Optional[Tuple[int, int]]
            'bandit_start_map': None,  # Optional[str]
            'bandit_total_movement': 0,  # int
            'last_map_for_bandit': None,  # Optional[str]

            # Interaction state
            'tried_interaction': False,  # bool
            'expected_interaction_tile': None,  # Optional[Tuple[int, int]]
            'interaction_failed': False,  # bool
            'interaction_stuck_steps': 0,  # int
            'reached_target_for_interaction': False,  # bool
            'last_actions': [],  # List[str]
            'movement_queue': [],  # List[str]
            'position_queue': [],  # List[Tuple[int, int]]
            'failed_movement_count': 0,  # int
            'skip_next_interaction': False,  # bool: Skip next A press if last was door/stairs
        }

        # Current state from game (updated each step via _update_state)
        self.player_map_tile_x: Optional[int] = None
        self.player_map_tile_y: Optional[int] = None
        self.n: Optional[str] = None
        self.player_facing: Optional[str] = None
        self.traversability_map: Optional[List[List[str]]] = None

        # NT features
        # Active tile cache tracking
        ensure_cache_directories()
        self.active_tile_index: Dict[str, Dict] = load_active_tile_index()

        # UCB Bandit exploration state (per map)
        self.coordinate_rewards: Dict[str, Dict[Tuple[int, int], float]] = {}  # map_name -> {(x,y): avg_reward}
        self.coordinate_visits: Dict[str, Dict[Tuple[int, int], int]] = {}  # map_name -> {(x,y): visit_count}
        self.global_step_counter: int = 1  # Global step counter for UCB calculation

        # Current frame for tile caching
        self.current_frame: Optional[np.ndarray] = None

        logger.info(f"Initialized NavigationExecutorNT: batch_size={action_batch_size}, "
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
        # Store current frame for tile caching
        self.current_frame = perception.frame

        # Update current state from game
        self._update_state(state_data)

        # Patch traversability map with active untraversable tiles
        self._patch_traversability_with_active_tiles()

        # Check for interaction results if we tried interaction last step
        if self.internal_state['tried_interaction']:
            self._check_interaction_result(state_data)

        if self.internal_state['phase'] == 'idle':
            return self._start_navigation(perception, state_data, goal)
        elif self.internal_state['phase'] == 'navigating':
            return self._execute_navigation(perception, state_data)
        elif self.internal_state['phase'] == 'interacting':
            return self._perform_interaction(perception, state_data)
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

            # NT features
            'failed_target_attempts': 0,
            'initial_distance_to_target': None,
            'displacement_history': [],
            'bandit_mode_active': False,
            'bandit_iterations': 0,
            'bandit_start_pos': None,
            'bandit_start_map': None,
            'bandit_total_movement': 0,
            'last_map_for_bandit': None,

            # Interaction state
            'tried_interaction': False,
            'expected_interaction_tile': None,
            'interaction_failed': False,
            'interaction_stuck_steps': 0,
            'reached_target_for_interaction': False,
            'last_actions': [],
            'movement_queue': [],
            'position_queue': [],
            'failed_movement_count': 0,
            'skip_next_interaction': False,
        }

    def _update_state(self, state_data: dict) -> None:
        """
        Extract and update state from state_data (player + map).

        Adapted from NavigationAgentNT._update_state.

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
        # Always regenerate base traversability map (without NPCs) on each step
        # This ensures the grid is centered on current player position
        # Active tiles will be patched in after this via _patch_traversability_with_active_tiles()
        map_data = state_data.get('map', {})
        self.traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)],
            npc_detections=None  # No NPCs for base grid
        )

        # Extract and cache portal connections from visual_map when map changes
        visual_map = state_data.get('visual_map', '')
        if visual_map and self.player_map and self.player_map != self.internal_state.get('last_map_for_bandit'):
            portal_connections = extract_portal_connections(visual_map)
            logger.info(f"Extracted {len(portal_connections)} portal connections from visual_map for {self.player_map}: {portal_connections}")
            if portal_connections:
                update_portal_connections_cache(portal_connections)
            else:
                # Clear cache if no portals found
                update_portal_connections_cache({})

    def _patch_traversability_with_active_tiles(self) -> None:
        """
        Patch self.traversability_map with active untraversable tiles from cache.

        Called each turn before pathfinding.
        Only applies tiles from the current map.
        """
        if not self.traversability_map:
            return

        if not self.player_map:
            return

        # Use the active_tile_index which has format: {filename: {'class': str, 'tile_pos': [x, y]}}
        if not self.active_tile_index:
            return

        try:
            # Iterate over active tiles
            for filename, tile_info in self.active_tile_index.items():
                tile_class = tile_info.get('class', '')

                # Only patch untraversable and NPC tiles
                if tile_class not in ['untraversable', 'npc']:
                    continue

                # Check if tile is from current map by parsing filename
                # Format: {map_name}_{x}_{y}.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) >= 3:
                    # Handle map names with underscores by taking all but last 2 parts
                    map_name = '_'.join(parts[:-2])

                    # Skip if not current map
                    if map_name != self.player_map:
                        continue

                tile_pos = tile_info.get('tile_pos', [])
                if len(tile_pos) == 2:
                    tile_x, tile_y = tile_pos

                    # Convert map tile to local grid position (player at 7,7)
                    if self.player_map_tile_x is not None and self.player_map_tile_y is not None:
                        local_x = 7 + (tile_x - self.player_map_tile_x)
                        local_y = 7 + (tile_y - self.player_map_tile_y)

                        # Check if tile is in current traversability map bounds
                        if 0 <= local_y < len(self.traversability_map) and 0 <= local_x < len(self.traversability_map[0]):
                            # Mark as untraversable
                            self.traversability_map[local_y][local_x] = '#'
                            logger.debug(f"Patched traversability map: tile ({tile_x}, {tile_y}) marked as untraversable (class={tile_class})")

        except Exception as e:
            logger.warning(f"Error patching traversability map with active tiles: {e}")

    def _start_navigation(
        self,
        perception: 'PerceptionResult',
        state_data: dict,
        goal: NavigationGoal
    ) -> ExecutorResult:
        """
        Start new navigation task.

        Adapted from NavigationAgentNT._start_new_navigation.

        Args:
            perception: Current perception result
            state_data: Game state data
            goal: NavigationTarget or dict containing target

        Returns:
            ExecutorResult with first action(s)
        """
        logger.info("Starting new navigation phase")

        # Handle clock objectives by generating and filtering targets
        if isinstance(goal, dict) and 'objective' in goal and 'clock' in goal['objective'].lower():
            logger.info("Clock objective detected - generating and filtering navigation targets")
            
            # Generate navigation targets using perception data
            targets = perception.navigation_targets
            if not targets:
                logger.warning("No navigation targets available for clock objective")
                return ExecutorResult(
                    actions=[],
                    status='failed',
                    summary="No navigation targets available for clock objective"
                )
            
            # Filter for clock targets if objective mentions clock
            original_count = len(targets)
            original_targets = targets  # Keep reference to original targets for passthrough
            
            # Debug: Log all target types before filtering
            target_types = {}
            for target in targets:
                target_types[target.type] = target_types.get(target.type, 0) + 1
            logger.info(f"Target types before clock filtering: {target_types}")
            
            clock_targets = [target for target in targets if target.type == "clock"]
            if len(clock_targets) > 0:
                targets = clock_targets
                filtered_count = original_count - len(targets)
                logger.info(f"Filtered for clock targets: kept {len(targets)} clock targets (filtered out {filtered_count})")
            else:
                # No clock targets, try stairs
                stair_targets = [target for target in targets if target.type == "stairs"]
                if len(stair_targets) > 0:
                    targets = stair_targets
                    filtered_count = original_count - len(targets)
                    logger.info(f"Filtered for stairs targets (fallback from clock): kept {len(targets)} stairs targets (filtered out {filtered_count})")
                else:
                    # No stairs targets either, passthrough all targets
                    targets = original_targets
                    logger.info(f"No clock or stairs targets found, using all {len(targets)} targets")
            
            # Manual check for "K" tiles if objective is "Fix clock" and on specific map
            if self.player_map == "LITTLEROOT TOWN BRENDANS HOUSE 2F":
                # Get player position for coordinate conversion
                player_x = self.player_map_tile_x
                player_y = self.player_map_tile_y
                player_pos = (player_x, player_y)
                
                # Collect all "K" tile positions from traversability map
                k_targets = []
                for y in range(15):
                    for x in range(15):
                        if self.traversability_map[y][x] == "K":
                            tile_dx = x - 7
                            tile_dy = y - 7
                            map_tile_x = player_x + tile_dx
                            map_tile_y = player_y + tile_dy
                            k_pos = (map_tile_x, map_tile_y)
                            
                            k_target = NavigationTarget(
                                id=f"k_tile_{map_tile_x}_{map_tile_y}",
                                type="clock",  # Treat as clock type
                                map_tile_position=k_pos,
                                local_tile_position=(x, y),
                                description=f"Clock tile at ({map_tile_x}, {map_tile_y})",
                                entity_type="clock_tile",
                                source_map_location=self.player_map
                            )
                            k_targets.append(k_target)
                
                if k_targets:
                    targets.extend(k_targets)
                    logger.info(f"Added {len(k_targets)} 'K' tile targets for clock objective")
                else:
                    logger.info("No 'K' tiles found for clock objective")
            
            # Choose the first target (simplified - could be enhanced with VLM selection)
            if targets:
                target = targets[0]
                logger.info(f"Selected clock target: {target.description}")
            else:
                logger.warning("No targets found after clock filtering")
                return ExecutorResult(
                    actions=[],
                    status='failed',
                    summary="No targets found after clock filtering"
                )
        else:
            # Extract target from goal (normal case)
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

        # Initialize displacement tracking
        if self.player_map_tile_x is not None and self.player_map_tile_y is not None:
            target_pos = target.map_tile_position
            current_distance = abs(self.player_map_tile_x - target_pos[0]) + abs(self.player_map_tile_y - target_pos[1])
            self.internal_state['initial_distance_to_target'] = current_distance
            self.internal_state['displacement_history'] = [current_distance]

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

        Adapted from NavigationAgentNT._execute_navigation.

        Args:
            perception: Current perception result
            state_data: Game state data

        Returns:
            ExecutorResult with next action(s)
        """
        # Check if reached target
        if self._has_reached_target(state_data):
            logger.info("Reached target, transitioning to interaction phase")

            # Reward the target coordinate for successful reach
            if self.internal_state['bandit_mode_active'] and self.internal_state['current_target']:
                target_coord = self.internal_state['current_target'].map_tile_position
                self._update_coordinate_reward(self.player_map, target_coord, reward=2.0)
                logger.debug(f"Bandit mode: rewarded target coordinate {target_coord} for being reached")

            self.internal_state['phase'] = 'interacting'
            self.internal_state['reached_target_for_interaction'] = True
            return ExecutorResult(actions=[], status='in_progress')

        # Track player state (map + position) to detect progress
        current_pos = (self.player_map_tile_x, self.player_map_tile_y)
        current_map = self.player_map

        # Initialize tracking on first iteration
        if self.internal_state['last_player_pos'] is None or self.internal_state['last_player_map'] is None:
            self.internal_state['last_player_pos'] = current_pos
            self.internal_state['last_player_map'] = current_map
            self.internal_state['last_map_for_bandit'] = current_map
            logger.debug(f"Initialized navigation tracking: map={current_map}, pos={current_pos}")

        position_changed = current_map != self.internal_state['last_player_map'] or current_pos != self.internal_state['last_player_pos']

        # Track displacement to target for bandit override detection
        if self.internal_state['current_target'] and current_pos[0] is not None and current_pos[1] is not None:
            target_pos = self.internal_state['current_target'].map_tile_position
            current_distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])

            # Initialize distance tracking on first measurement
            if self.internal_state['initial_distance_to_target'] is None:
                self.internal_state['initial_distance_to_target'] = current_distance
                self.internal_state['displacement_history'] = [current_distance]
            else:
                # Add to displacement history (keep last 5)
                self.internal_state['displacement_history'].append(current_distance)
                if len(self.internal_state['displacement_history']) > 5:
                    self.internal_state['displacement_history'].pop(0)

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
            self.internal_state['last_map_for_bandit'] = current_map

            # Update coordinate reward (positive for progress)
            if self.internal_state['bandit_mode_active']:
                self._update_coordinate_reward(current_map, current_pos, reward=1.0)
                logger.debug(f"Bandit mode: rewarded coordinate {current_pos} for progress")

            # Check if displacement improved sufficiently (>30% reduction)
            if (self.internal_state['initial_distance_to_target'] is not None and
                len(self.internal_state['displacement_history']) >= 2):
                initial_dist = self.internal_state['displacement_history'][0]
                current_dist = self.internal_state['displacement_history'][-1]
                if initial_dist > 0:
                    improvement = (initial_dist - current_dist) / initial_dist
                    if improvement < 0.3:  # Less than 30% improvement
                        logger.warning(f"Insufficient progress: only {improvement*100:.1f}% improvement toward target")
        else:
            # No progress (map AND position unchanged)
            self.internal_state['steps_since_progress'] += 1
            logger.warning(f"No progress: {self.internal_state['steps_since_progress']}/{self.stuck_threshold}")

            # Only increment failed attempts if we're genuinely stuck (at threshold)
            if self.internal_state['steps_since_progress'] >= self.stuck_threshold:
                self.internal_state['failed_target_attempts'] += 1
                logger.warning(f"Stuck at threshold, failed_target_attempts: {self.internal_state['failed_target_attempts']}")

        # Check if bandit mode should be activated
        if not self.internal_state['bandit_mode_active'] and self._should_activate_bandit_override():
            logger.warning("🎰 Activating UCB bandit exploration override")
            self.internal_state['bandit_mode_active'] = True
            self.internal_state['bandit_iterations'] = 0
            self.internal_state['bandit_start_pos'] = current_pos
            self.internal_state['bandit_start_map'] = current_map
            self.internal_state['bandit_total_movement'] = 0

        # Check if bandit mode should be deactivated
        if self.internal_state['bandit_mode_active'] and self._should_deactivate_bandit_mode(current_pos, current_map):
            logger.info("🎯 Resuming normal navigation")
            self.internal_state['bandit_mode_active'] = False
            self.internal_state['bandit_iterations'] = 0
            self.internal_state['bandit_start_pos'] = None
            self.internal_state['bandit_start_map'] = None
            self.internal_state['bandit_total_movement'] = 0

        # Check if stuck (no progress for threshold steps)
        if self.internal_state['steps_since_progress'] >= self.stuck_threshold:
            # Penalize the target coordinate for being unreachable
            if self.internal_state['bandit_mode_active'] and self.internal_state['current_target']:
                target_coord = self.internal_state['current_target'].map_tile_position
                self._update_coordinate_reward(current_map, target_coord, reward=-1.0)
                logger.debug(f"Bandit mode: penalized target coordinate {target_coord} for being unreachable")

            logger.warning(f"Stuck for {self.internal_state['steps_since_progress']} steps - transitioning to interacting phase")
            self.internal_state['phase'] = 'interacting'
            self.internal_state['reached_target_for_interaction'] = False
            return self._perform_interaction(perception, state_data)

        # Check if path is exhausted
        if self.internal_state['path_index'] >= len(self.internal_state['current_path']):
            # Penalize the target coordinate for path exhaustion
            if self.internal_state['bandit_mode_active'] and self.internal_state['current_target']:
                target_coord = self.internal_state['current_target'].map_tile_position
                self._update_coordinate_reward(current_map, target_coord, reward=-0.5)
                logger.debug(f"Bandit mode: penalized target coordinate {target_coord} for path exhaustion")

            logger.warning("Path exhausted but target not reached - transitioning to interacting phase")
            self.internal_state['phase'] = 'interacting'
            self.internal_state['reached_target_for_interaction'] = False
            return self._perform_interaction(perception, state_data)

        # Emit next batch of actions
        return self._emit_next_actions()

    def _perform_interaction(
        self,
        perception: 'PerceptionResult',
        state_data: dict
    ) -> ExecutorResult:
        """
        Phase 3: Handle interaction with reached target OR stuck situation.

        For reached targets: Wait to see if stuck, then press 'A' if stuck.
        For stuck situations: Press 'A' immediately.

        Args:
            perception: Current perception result
            state_data: Game state data

        Returns:
            ExecutorResult with interaction actions or empty to trigger new navigation
        """
        # If we should skip the next interaction (e.g., last was door/stairs), skip it
        if self.internal_state['skip_next_interaction']:
            self.internal_state['skip_next_interaction'] = False
            logger.info("Skipping interaction as last one was door/stairs")
            self.internal_state['phase'] = 'idle'
            return ExecutorResult(actions=[], status='completed')

        # Calculate current distance to target
        current_target = self.internal_state['current_target']
        player_pos = (self.player_map_tile_x, self.player_map_tile_y)
        target_pos = current_target.map_tile_position if current_target else None
        distance = abs(player_pos[0] - target_pos[0]) + abs(player_pos[1] - target_pos[1]) if target_pos else float('inf')

        # Check if map changed
        current_map = self.player_map
        target_source_map = current_target.source_map_location if current_target else None
        map_changed = (target_source_map is not None and
                      current_map is not None and
                      current_map != target_source_map)

        # Handle reached target
        if self.internal_state['reached_target_for_interaction']:
            if map_changed or distance == 0:
                logger.info("Target reached successfully")
                self.internal_state['phase'] = 'idle'
                return ExecutorResult(actions=[], status='completed')
            elif distance == 1:
                logger.info(f"Adjacent to target (distance={distance}) - pressing 'A' to interact")
                self.internal_state['tried_interaction'] = True
                self.internal_state['expected_interaction_tile'] = target_pos
                return ExecutorResult(actions=['A'], status='in_progress')
            else:
                # Distance >1, should not happen if reached
                logger.warning(f"Reached target but distance={distance} >1, resetting")
                self.internal_state['phase'] = 'idle'
                return ExecutorResult(actions=[], status='completed')

        # Handle stuck situations (path exhausted or movement stuck)
        else:
            if current_target and current_target.type == "object" and distance == 1:
                logger.info(f"Stuck situation, adjacent to object target (distance={distance}) - pressing 'A' to interact")
                self.internal_state['tried_interaction'] = True
                self.internal_state['expected_interaction_tile'] = target_pos
                return ExecutorResult(actions=['A'], status='in_progress')
            else:
                # Check if distance > 1 and we haven't tried interaction yet
                if distance > 1 and not self.internal_state['tried_interaction']:
                    attempted_tile = self._get_attempted_interaction_tile(player_pos[0], player_pos[1])
                    if attempted_tile:
                        logger.info(f"Stuck with distance >1, trying to interact once with blocking transition tile {attempted_tile} to check if NPC or obstacle")
                        self.internal_state['tried_interaction'] = True
                        self.internal_state['expected_interaction_tile'] = attempted_tile
                        return ExecutorResult(actions=['A'], status='in_progress')
                    else:
                        logger.warning("Could not determine attempted tile, resetting")
                        self.internal_state['phase'] = 'idle'
                        return ExecutorResult(actions=[], status='completed')
                else:
                    # Not adjacent to object or not an object target, or already tried interaction, reset
                    logger.info(f"Stuck but not adjacent to object target (distance={distance}, type={current_target.type if current_target else None}), or already tried interaction, resetting")
                    self.internal_state['phase'] = 'idle'
                    return ExecutorResult(actions=[], status='completed')

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

            # Track movement for stuck detection
            for action in actions_to_emit:
                if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    current_pos = (self.player_map_tile_x, self.player_map_tile_y)
                    self._track_movement_queue(action, current_pos)
                    self._check_failed_movement(action, current_pos, self.current_frame)

        return ExecutorResult(actions=actions_to_emit, status='in_progress')

    def _update_probe_result(self, position_changed: bool) -> None:
        """
        Process probe result to determine facing direction.

        Adapted from NavigationAgentNT._update_probe_result.

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

        Always plans path to step ON the target coordinate.

        Adapted from NavigationAgentNT._plan_path_to_target.

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

        Adapted from NavigationAgentNT._has_reached_target.

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

    def _get_facing_action_for_target(self) -> Optional[str]:
        """
        Get action to face the current target.

        Returns:
            Action string or None if already facing or can't determine
        """
        if not self.internal_state['current_target'] or not self.player_facing:
            return None

        target_pos = self.internal_state['current_target'].map_tile_position
        player_pos = (self.player_map_tile_x, self.player_map_tile_y)

        # Calculate direction to target
        dx = target_pos[0] - player_pos[0]
        dy = target_pos[1] - player_pos[1]

        # Determine required facing direction
        if abs(dx) > abs(dy):
            required_facing = "East" if dx > 0 else "West"
        else:
            required_facing = "South" if dy > 0 else "North"

        if self.player_facing == required_facing:
            return None  # Already facing correctly

        # Map facing to turn actions
        facing_to_action = {
            "North": {"East": "RIGHT", "West": "LEFT"},
            "South": {"West": "RIGHT", "East": "LEFT"},
            "East": {"South": "RIGHT", "North": "LEFT"},
            "West": {"North": "RIGHT", "South": "LEFT"}
        }

        return facing_to_action.get(self.player_facing, {}).get(required_facing)

    def _get_attempted_interaction_tile(self, player_x: int, player_y: int) -> Optional[Tuple[int, int]]:
        """
        Determine the tile we attempted to interact with or move to.

        Priority order:
        1. Expected interaction tile (set when pressing 'A')
        2. Current target if adjacent
        3. From last movement action in queue
        4. From current path

        Args:
            player_x: Current player X position
            player_y: Current player Y position

        Returns:
            (x, y) tuple of attempted tile, or None if cannot determine
        """
        # 1. First check if we stored an expected interaction tile (highest priority)
        if self.internal_state['expected_interaction_tile'] is not None:
            return self.internal_state['expected_interaction_tile']

        # 2. Check if we have a current target and are adjacent to it
        if self.internal_state['current_target']:
            target_pos = self.internal_state['current_target'].map_tile_position
            distance = abs(player_x - target_pos[0]) + abs(player_y - target_pos[1])
            if distance == 1:
                return target_pos

        # 3. Try to determine from last movement action in queue (most recent movement)
        if self.internal_state['movement_queue']:
            # Get most recent movement action
            for action in reversed(self.internal_state['movement_queue']):
                if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                    last_movement_action = action
                    break
            else:
                last_movement_action = None

            if last_movement_action:
                action_offset = {
                    'UP': (0, -1),
                    'DOWN': (0, 1),
                    'LEFT': (-1, 0),
                    'RIGHT': (1, 0)
                }
                offset = action_offset.get(last_movement_action, (0, 0))
                next_tile = (player_x + offset[0], player_y + offset[1])
                return next_tile

        # 4. Try to determine from current path and path_index
        if self.internal_state['current_path'] and self.internal_state['path_index'] > 0:
            last_action_index = self.internal_state['path_index'] - 1
            if last_action_index < len(self.internal_state['current_path']):
                last_action = self.internal_state['current_path'][last_action_index]

                action_offset = {
                    'UP': (0, -1),
                    'DOWN': (0, 1),
                    'LEFT': (-1, 0),
                    'RIGHT': (1, 0)
                }

                if last_action in action_offset:
                    offset = action_offset[last_action]
                    next_tile = (player_x + offset[0], player_y + offset[1])
                    return next_tile

        return None

    def _check_interaction_result(self, state_data: dict) -> None:
        """
        Check the result of pressing 'A' button in previous step.

        Handles different interaction outcomes:
        - If map changed: tile was door/stairs, erase from cache
        - If no map change: assume untraversable (since executor doesn't have frame for dialogue detection)

        This is called at the START of the next step after pressing 'A'.

        Args:
            state_data: Game state data
        """
        # Get player position and map
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        player_x = position.get('x')
        player_y = position.get('y')
        player_map = player_data.get('location')

        # Get the tile we tried to interact with
        interaction_tile = self.internal_state['expected_interaction_tile']
        if not interaction_tile:
            logger.warning("No expected interaction tile set")
            self._reset_interaction_state()
            self.internal_state['interaction_failed'] = True
            self.internal_state['phase'] = 'idle'  # Reset phase if no tile set
            return

        # Check if map changed (indicates door/stairs)
        map_changed = (self.internal_state['last_player_map'] and
                      player_map != self.internal_state['last_player_map'])

        if map_changed:
            # Map changed - this was a door/stairs, erase from cache
            logger.info(f"Map changed after 'A' press - tile {interaction_tile} was door/stairs, erasing from cache")

            tile_filename = f"{self.internal_state['last_player_map']}_{interaction_tile[0]}_{interaction_tile[1]}.png"
            if tile_filename in self.active_tile_index:
                del self.active_tile_index[tile_filename]
                save_active_tile_index(self.active_tile_index)
                logger.debug(f"Erased {tile_filename} from active_tile_index")

            # Reset and continue (successful door/stairs interaction)
            self._reset_interaction_state()
            self.internal_state['interaction_failed'] = False
            self.internal_state['phase'] = 'idle'  # Transition back to idle after successful interaction
            self.internal_state['skip_next_interaction'] = True  # Skip any subsequent A press
            return

        # No map change - since we don't have frame/dialogue detection, assume untraversable
        # But if it's the target tile, don't mark as untraversable (it's obviously traversable)
        current_target = self.internal_state['current_target']
        target_pos = current_target.map_tile_position if current_target else None
        if target_pos and interaction_tile == target_pos:
            logger.info(f"No map change after 'A' press on target tile {interaction_tile} - not marking as untraversable")
            # Reset and start new navigation
            self._reset_interaction_state()
            self.internal_state['interaction_failed'] = True
            self.internal_state['phase'] = 'idle'
            return

        logger.info(f"No map change after 'A' press - updating tile {interaction_tile} to untraversable")

        tile_filename = f"{player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
        if tile_filename in self.active_tile_index:
            self.active_tile_index[tile_filename]['class'] = 'untraversable'
            save_active_tile_index(self.active_tile_index)
            logger.debug(f"Updated {tile_filename} class to untraversable")
        else:
            # Mark as untraversable
            mark_and_save_tile(
                interaction_tile, None, 'untraversable', self.active_tile_index, player_map,
                allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                player_map_tile_y=self.player_map_tile_y
            )

        log_interaction(
            action='A',
            tile_pos=interaction_tile,
            result='untraversable_non_interactable',
            is_npc=False,
            map_location=player_map,
            navigation_target=None
        )

        # Reset and start new navigation
        self._reset_interaction_state()
        self.internal_state['interaction_failed'] = True
        self.internal_state['phase'] = 'idle'  # Transition back to idle after handling interaction

    def _reset_interaction_state(self) -> None:
        """Reset interaction-related state."""
        self.internal_state['tried_interaction'] = False
        self.internal_state['expected_interaction_tile'] = None
        self.internal_state['interaction_stuck_steps'] = 0
        self.internal_state['reached_target_for_interaction'] = False
        self.internal_state['skip_next_interaction'] = False

    def _track_movement_queue(self, action: str, current_pos: Tuple[int, int]) -> None:
        """
        Track movement actions and positions for stuck detection.

        Args:
            action: Action taken
            current_pos: Current (x, y) position
        """
        # Only track directional movements
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            self.internal_state['movement_queue'].append(action)
            self.internal_state['position_queue'].append(current_pos)

            # Keep only last 3
            if len(self.internal_state['movement_queue']) > 3:
                self.internal_state['movement_queue'].pop(0)
                self.internal_state['position_queue'].pop(0)

    def _check_failed_movement(self, action: str, current_pos: Tuple[int, int], frame: Optional[np.ndarray] = None) -> None:
        """
        Check if movement has failed 3 consecutive times.

        If same action 3 times with no position change, mark the blocked tile.

        Args:
            action: Action taken
            current_pos: Current (x, y) position
            frame: Game frame for tile caching (optional)
        """
        # Check if we have 3 consecutive same actions with no position change
        if (len(self.internal_state['movement_queue']) >= 3 and
            len(self.internal_state['position_queue']) >= 3):

            # Check if last 3 actions are the same
            last_3_actions = self.internal_state['movement_queue'][-3:]
            if len(set(last_3_actions)) == 1:  # All same action
                # Check if position didn't change over last 3 steps
                last_3_positions = self.internal_state['position_queue'][-3:]
                if len(set(last_3_positions)) == 1:  # Position didn't change
                    # Failed movement detected
                    self.internal_state['failed_movement_count'] += 1
                    logger.warning(f"Failed movement detected: {action} x3 with no position change (count: {self.internal_state['failed_movement_count']})")

                    # Mark the attempted tile as untraversable
                    action_offset = {
                        'UP': (0, -1),
                        'DOWN': (0, 1),
                        'LEFT': (-1, 0),
                        'RIGHT': (1, 0)
                    }
                    offset = action_offset.get(action, (0, 0))
                    blocked_tile = (current_pos[0] + offset[0], current_pos[1] + offset[1])

                    tile_filename = f"{self.player_map}_{blocked_tile[0]}_{blocked_tile[1]}.png"
                    if tile_filename in self.active_tile_index:
                        self.active_tile_index[tile_filename]['class'] = 'untraversable'
                        save_active_tile_index(self.active_tile_index)
                        logger.debug(f"Updated {tile_filename} class to untraversable due to failed movement")
                    else:
                        # Mark tile as untraversable in cache (only if frame is available)
                        if frame is not None:
                            mark_and_save_tile(
                                blocked_tile, frame, 'untraversable', self.active_tile_index, self.player_map,
                                allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                                player_map_tile_y=self.player_map_tile_y
                            )
                        else:
                            # Update existing tile class if already in cache
                            tile_filename = f"{self.player_map}_{blocked_tile[0]}_{blocked_tile[1]}.png"
                            if tile_filename in self.active_tile_index:
                                self.active_tile_index[tile_filename]['class'] = 'untraversable'
                                save_active_tile_index(self.active_tile_index)
                                logger.debug(f"Updated {tile_filename} class to untraversable due to failed movement (no frame)")
                            else:
                                logger.debug(f"Cannot mark tile {blocked_tile} as untraversable - no frame available and not in cache")

                    log_interaction(
                        action=action,
                        tile_pos=blocked_tile,
                        result='untraversable_failed_movement',
                        is_npc=False,
                        map_location=self.player_map,
                        navigation_target=None
                    )

    def _should_activate_bandit_override(self) -> bool:
        """
        Check if bandit exploration override should be activated.

        Activates when:
        - 5+ failed attempts to reach target
        - Displacement has not improved by at least 30%

        Returns:
            True if bandit mode should be activated
        """
        if self.internal_state['failed_target_attempts'] < 5:
            return False

        if (self.internal_state['initial_distance_to_target'] is None or
            len(self.internal_state['displacement_history']) < 5):
            return False

        initial_dist = self.internal_state['displacement_history'][0]
        current_dist = self.internal_state['displacement_history'][-1]

        if initial_dist == 0:
            return False  # Already at target

        improvement = (initial_dist - current_dist) / initial_dist

        if improvement < 0.3:
            logger.warning(f"Activating bandit override: only {improvement*100:.1f}% improvement after 5 attempts")
            return True

        return False

    def _should_deactivate_bandit_mode(self, current_pos: Tuple[int, int], current_map: str) -> bool:
        """
        Check if bandit mode should be deactivated.

        Deactivates when:
        - 10+ bandit iterations completed
        - Total movement from bandit start > 10 tiles (across map transitions)

        Args:
            current_pos: Current player position (x, y)
            current_map: Current map name

        Returns:
            True if should resume normal navigation
        """
        if self.internal_state['bandit_iterations'] >= 10:
            logger.info("Deactivating bandit mode: 10 iterations completed")
            return True

        # Track cumulative movement across map transitions
        if (self.internal_state['bandit_start_pos'] is not None and
            self.internal_state['bandit_start_map'] is not None):

            # Update total movement if position or map changed
            if self.internal_state['last_map_for_bandit'] != current_map:
                # Map changed - count as 1 tile movement
                self.internal_state['bandit_total_movement'] += 1
                logger.debug(f"Bandit map transition: {self.internal_state['last_map_for_bandit']} -> {current_map}, +1 movement")
            elif self.internal_state['last_player_pos'] != current_pos:
                # Position changed on same map - add Manhattan distance
                movement = abs(current_pos[0] - self.internal_state['last_player_pos'][0]) + \
                          abs(current_pos[1] - self.internal_state['last_player_pos'][1])
                self.internal_state['bandit_total_movement'] += movement

            # Update tracking
            self.internal_state['last_map_for_bandit'] = current_map

            if self.internal_state['bandit_total_movement'] > 10:
                logger.info(f"Deactivating bandit mode: moved {self.internal_state['bandit_total_movement']} tiles from start")
                return True

        return False

    def _initialize_coordinate_reward(self, map_name: str, coord: Tuple[int, int], default_reward: float = 0.0) -> float:
        """
        Initialize reward for a coordinate by checking nearby visited coordinates.

        If a nearby coordinate (within 3 tiles) has a valid reward, use it as initialization.
        Otherwise use default_reward.

        Args:
            map_name: Current map name
            coord: Coordinate to initialize (x, y)
            default_reward: Default reward if no nearby coordinates found

        Returns:
            Initial reward value
        """
        if map_name not in self.coordinate_rewards:
            self.coordinate_rewards[map_name] = {}
        if map_name not in self.coordinate_visits:
            self.coordinate_visits[map_name] = {}

        x, y = coord
        # Check neighbors within 3 tiles (Manhattan distance)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in self.coordinate_rewards[map_name]:
                    # Found a nearby visited coordinate, use its reward
                    nearby_reward = self.coordinate_rewards[map_name][neighbor]
                    logger.debug(f"Initializing {coord} reward from nearby {neighbor}: {nearby_reward:.3f}")
                    initial_reward = nearby_reward
                    break
        else:
            initial_reward = default_reward

        # Initialize the coordinate in both dictionaries
        self.coordinate_rewards[map_name][coord] = initial_reward
        self.coordinate_visits[map_name][coord] = 0

        return initial_reward

    def _update_coordinate_reward(self, map_name: str, coord: Tuple[int, int], reward: float) -> None:
        """
        Update the average reward for a coordinate using incremental average.

        Args:
            map_name: Current map name
            coord: Coordinate (x, y)
            reward: Observed reward (1.0 for progress, 0.0 for blocked)
        """
        # Ensure coordinate is initialized in both dictionaries
        self._initialize_coordinate_reward(map_name, coord)

        # Update visit count
        self.coordinate_visits[map_name][coord] += 1
        n = self.coordinate_visits[map_name][coord]

        # Update average reward incrementally: Q_n+1 = Q_n + (R - Q_n) / (n+1)
        old_q = self.coordinate_rewards[map_name][coord]
        self.coordinate_rewards[map_name][coord] = old_q + (reward - old_q) / n

        logger.debug(f"Updated reward for {coord} on {map_name}: {self.coordinate_rewards[map_name][coord]:.3f} (n={n})")

    def _compute_ucb(self, map_name: str, coord: Tuple[int, int]) -> float:
        """
        Compute Upper Confidence Bound for a coordinate.

        UCB = Q(coord) + sqrt(2 * ln(t) / N(coord))

        Args:
            map_name: Current map name
            coord: Coordinate (x, y)

        Returns:
            UCB value (infinity for unvisited coordinates)
        """
        visits_dict = self.coordinate_visits.get(map_name, {})
        if coord not in visits_dict or visits_dict[coord] == 0:
            return float('inf')  # Force exploration of new coordinates

        q_value = self.coordinate_rewards.get(map_name, {}).get(coord, 0.0)
        n_visits = visits_dict[coord]

        exploration_bonus = np.sqrt(2 * np.log(self.global_step_counter) / n_visits)
        ucb = q_value + exploration_bonus

        return ucb

    def _select_bandit_target(self, current_pos: Tuple[int, int], current_map: str) -> Optional[NavigationTarget]:
        """
        Select a target using UCB bandit strategy for exploration.

        Since executor doesn't have all targets, this selects coordinates to explore.

        Args:
            current_pos: Current player position (x, y)
            current_map: Current map name

        Returns:
            NavigationTarget for exploration or None
        """
        # For bandit mode, we need to select coordinates to explore
        # Since we don't have all targets, we'll select high-UCB coordinates around current position

        # Get candidate coordinates within exploration range
        candidates = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if abs(dx) + abs(dy) > 5:  # Limit to Manhattan distance 5
                    continue
                coord = (current_pos[0] + dx, current_pos[1] + dy)
                if coord == current_pos:
                    continue

                # Initialize and get UCB
                self._initialize_coordinate_reward(current_map, coord)
                ucb = self._compute_ucb(current_map, coord)
                candidates.append((ucb, coord))

        if not candidates:
            return None

        # Select coordinate with maximum UCB
        _, selected_coord = max(candidates, key=lambda x: x[0])

        # Create a boundary-like target for exploration
        # Convert to local tile position relative to player
        local_x = 7 + (selected_coord[0] - current_pos[0])
        local_y = 7 + (selected_coord[1] - current_pos[1])

        # Check if within traversability map bounds
        if not (0 <= local_y < len(self.traversability_map) and 0 <= local_x < len(self.traversability_map[0])):
            return None

        # Check if traversable
        if self.traversability_map[local_y][local_x] == '#':
            return None  # Untraversable

        exploration_target = NavigationTarget(
            id=f"bandit_exploration_{selected_coord[0]}_{selected_coord[1]}",
            type="boundary",  # Treat as boundary for exploration
            map_tile_position=selected_coord,
            local_tile_position=(local_x, local_y),
            description=f"Bandit exploration target at {selected_coord}",
            entity_type="exploration",
            source_map_location=current_map
        )

        logger.info(f"Bandit selected exploration target: {selected_coord} (UCB: {max(candidates, key=lambda x: x[0])[0]:.3f})")
        return exploration_target