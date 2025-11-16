"""
Navigation agent with object detection, A* pathfinding, and VLM decision making.

WIP: this file is under construction

TODO: FUTURE: split into subagents.
"""
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
import logging

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent
# from custom_utils.object_detector import ObjectDetector
from custom_utils.cached_object_detector import CachedObjectDetector as ObjectDetector
from custom_utils.navigation_targets import (
    NavigationTarget, 
    generate_navigation_targets
)
from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid, add_turn_actions
from custom_utils.map_extractor import get_player_centered_grid

logger = logging.getLogger(__name__)


# Prompt templates
TARGET_SELECTION_PROMPT = """
You are playing Pokemon Emerald. Choose where to navigate.

DETECTED TARGETS:
{formatted_targets}

You are the person at ({player_map_tile_x}, {player_map_tile_y}) of size 1x1.
The screen is of size 15x10.

Choose the target that best advances your goal. 
You must prioritize interacting with characters.
"""

INTERACTION_PROMPT = """
You reached: {target_description}
Current situation: {state_summary}

Decide interaction. Common patterns:
- 'A' to talk/interact/open
- 'A, A, A' for dialogue
- 'B' to cancel
"""


class NavigationState(BaseModel):
    """Track navigation state across agent steps."""
    phase: Literal["idle", "choosing_target", "navigating", "interacting"]
    current_target: Optional[NavigationTarget] = None
    current_path: List[str] = []
    path_index: int = 0
    steps_since_progress: int = 0
    last_player_pos: Optional[tuple] = None  # (x, y)
    last_player_map: Optional[str] = None  # map location
    facing_verified: bool = False  # Have we verified facing direction?
    awaiting_probe_result: bool = False  # Waiting for probe result?
    naive_path: List[str] = []  # Store original naive path before turn processing


class TargetDecision(BaseModel):
    """VLM structured output for target selection."""
    reasoning: str = Field(description="Reasoning for target choice")
    target_index: int = Field(description="Index of chosen target (0-based)")


class InteractionDecision(BaseModel):
    """VLM structured output for interaction decision."""
    reasoning: str = Field(description="Reasoning for interaction")
    button_names: List[Literal['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'L', 'R']] = Field(
        description="Buttons to press"
    )


@AgentRegistry.register("navigation")
class NavigationAgent(BaseLangChainAgent):
    """Navigation agent with object detection and A* pathfinding."""
    
    def __init__(
        self, 
        backend: str = "github_models", 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        action_batch_size: int = 100,
        stuck_threshold: int = 3,
        movement_mode: Literal["naive", "facing_aware"] = "naive",
        **kwargs
    ):
        """
        Initialize navigation agent.
        
        Args:
            backend: LLM backend type
            model_name: Model name
            temperature: Generation temperature
            action_batch_size: Number of actions to emit per step (default 1)
            stuck_threshold: Steps without progress before replanning (default 3)
            movement_mode: Movement mode ("naive" or "facing_aware")
            **kwargs: Additional args passed to BaseLangChainAgent
        """
        super().__init__(
            backend=backend, 
            model_name=model_name, 
            temperature=temperature, 
            **kwargs
        )
        
        self.object_detector = ObjectDetector()
        self.pathfinder = AStarPathfinder()
        self.nav_state = NavigationState(phase="idle")
        
        self.action_batch_size = action_batch_size
        self.stuck_threshold = stuck_threshold
        self.movement_mode = movement_mode
        
        # Current state (updated each step)
        self.player_map_tile_x: Optional[int] = None
        self.player_map_tile_y: Optional[int] = None
        self.player_map: Optional[str] = None
        # Facing is inferred via probe-and-process: first action acts as probe
        self.player_facing: Optional[str] = None
        self.traversability_map: Optional[List[List[str]]] = None
        
        logger.info(f"Initialized NavigationAgent with {backend}/{model_name}")
        logger.info(f"Config: batch_size={action_batch_size}, stuck_threshold={stuck_threshold}, movement_mode={movement_mode}")
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> List[str]:
        """
        Main decision loop with three-phase navigation.
        
        Args:
            state_data: Game state data from emulator
            frame: Game frame as numpy array
            **kwargs: Additional arguments
            
        Returns:
            List of button actions to execute
        """
        # Update state (player + map) first
        self._update_state(state_data)
        
        if self.nav_state.phase in ["idle", "choosing_target"]:
            return self._start_new_navigation(state_data, frame)
        
        elif self.nav_state.phase == "navigating":
            return self._execute_navigation(state_data, frame)
        
        elif self.nav_state.phase == "interacting":
            return self._perform_interaction(state_data, frame)
    
    def _update_state(self, state_data: dict) -> None:
        """
        Extract and update state from state_data (player + map).
        
        Raises warnings if extraction fails (no default values).
        
        Args:
            state_data: Game state data from emulator
        """
        player_data: dict = state_data.get('player', {})
        
        # Extract position (nested in position dict)
        position = player_data.get('position')
        if position is None or not isinstance(position, dict):
            logger.warning(f"Failed to extract player position from state_data. player_data keys: {list(player_data.keys())}")
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
            logger.warning(f"Failed to extract player location from state_data. player_data keys: {list(player_data.keys())}")
        
        # Note: Facing direction is inferred during navigation via probe-and-process
        # (first action of path acts as a probe to determine facing)
        
        # Extract traversability map from visual_map
        map_data = state_data.get('map', {})
        self.traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)]
        )
    
    def _start_new_navigation(self, state_data: dict, frame: np.ndarray) -> List[str]:
        """
        Phase 1: Detect objects, generate targets, choose with VLM, plan naive path.
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            First action (used as probe to determine facing)
        """
        logger.info("Starting new navigation phase")
        
        # Step 1: Detect objects
        logger.info("Detecting objects in frame")
        detected_objects = self.object_detector.detect_objects(frame)
        logger.info(f"Detected {len(detected_objects)} objects")
        
        # Step 2: Generate navigation targets
        player_map_tile_pos = (self.player_map_tile_x, self.player_map_tile_y)
        
        logger.info("Generating navigation targets")
        targets = generate_navigation_targets(
            detected_objects=detected_objects,
            traversability_map=self.traversability_map,
            player_map_tile_pos=player_map_tile_pos,
            player_map_location=self.player_map
        )
        logger.info(f"Generated {len(targets)} navigation targets")
        
        if not targets:
            logger.warning("No targets found, returning empty action")
            self.nav_state.phase = "idle"
            return []
        
        # Step 3: Choose target with VLM
        chosen_target = self._choose_target_with_vlm(targets, frame, state_data)
        
        # Step 4: Plan path to target
        logger.info(f"Planning path to target: {chosen_target.description}")
        path = self._plan_path_to_target(chosen_target, state_data)
        
        # Append exit step for doors (after planning, before execution)
        if chosen_target.type == "door" and chosen_target.exit_direction:
            exit_action = self._direction_to_action(chosen_target.exit_direction)
            path.append(exit_action)
            logger.info(f"Appended exit step '{exit_action}' for door")

        # Update navigation state
        self.nav_state.phase = "navigating"
        self.nav_state.current_target = chosen_target
        self.nav_state.path_index = 0
        self.nav_state.steps_since_progress = 0
        self.nav_state.last_player_pos = player_map_tile_pos
        self.nav_state.last_player_map = self.player_map

        if not path:
            logger.warning("No path found")
            return []

        # Setup probe-and-process for facing_aware mode
        if self.movement_mode == "facing_aware" and not self.nav_state.facing_verified:
            self.nav_state.naive_path = path
            self.nav_state.awaiting_probe_result = True
            self.nav_state.current_path = []  # Empty until probe is processed
            logger.info(f"Facing-aware mode: emitting first action '{path[0]}' as probe")
            return [path[0]]  # Return probe directly
        else:
            # Normal mode or facing already verified
            self.nav_state.naive_path = []
            self.nav_state.awaiting_probe_result = False
            self.nav_state.current_path = path
            logger.info(f"Path planned with {len(path)} actions")
            return self._emit_next_actions()  # Normal emission
    
    def _update_probe_result(self, position_changed: bool) -> None:
        logger.info("Processing probe result to determine facing direction")
        
        # Check if player moved (position changed)
        probe_action = self.nav_state.naive_path[0]
        
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
            remaining_naive_path = self.nav_state.naive_path[1:]
        else:
            # Player just turned - was NOT facing that direction
            logger.info(f"Probe result: Position unchanged. Player turned to face {inferred_facing}")
            # First action was turn, not consumed from naive path
            remaining_naive_path = self.nav_state.naive_path
        
        # Update player facing
        self.player_facing = inferred_facing
        self.nav_state.facing_verified = True
        
        # Process remaining naive path with verified facing
        processed_path = add_turn_actions(remaining_naive_path, inferred_facing)
        
        # Update path and clear probe state
        self.nav_state.current_path = processed_path
        self.nav_state.path_index = 0
        self.nav_state.awaiting_probe_result = False
        
        logger.info(f"Processed remaining path: {len(processed_path)} actions")
        
    def _reset_nav_state(self) -> None:
        self.nav_state.phase = "idle"
        self.nav_state.last_player_pos = None
        self.nav_state.last_player_map = None
        self.nav_state.facing_verified = False  # Reset for next navigation

    def _execute_navigation(self, state_data: dict, frame: np.ndarray) -> List[str]:
        """
        Phase 2: Execute path step-by-step, check for arrival or stuck.
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            Next action(s) in path
        """
        # Check if reached target
        if self._has_reached_target(state_data):
            logger.info("Reached target, transitioning to interaction phase")
            self.nav_state.phase = "interacting"
            return []
        
        # Track player state (map + position) to detect progress
        current_pos = (self.player_map_tile_x, self.player_map_tile_y)
        current_map = self.player_map

        position_changed = current_map != self.nav_state.last_player_map or current_pos != self.nav_state.last_player_pos

        # Check if awaiting probe result
        if self.nav_state.awaiting_probe_result:
            self._update_probe_result(position_changed)
        
        # Check if state changed (progress made)
        # Reason: Use elif to avoid checking on first initialization
        # if self.nav_state.last_player_pos is None or self.nav_state.last_player_map is None:
        #     # First time, just initialize without checking progress
        #     self.nav_state.last_player_pos = current_pos
        #     self.nav_state.last_player_map = current_map
        #     logger.debug(f"Initialized tracking: map={current_map}, pos={current_pos}")
        # elif current_map != self.nav_state.last_player_map or current_pos != self.nav_state.last_player_pos:
        if position_changed:
            # Progress detected (map changed OR position changed)
            if current_map != self.nav_state.last_player_map:
                logger.info(f"Progress: Map changed from '{self.nav_state.last_player_map}' to '{current_map}'")
            else:
                logger.debug(f"Progress: Position changed from {self.nav_state.last_player_pos} to {current_pos}")
            
            self.nav_state.steps_since_progress = 0
            self.nav_state.last_player_pos = current_pos
            self.nav_state.last_player_map = current_map
        else:
            # No progress (map AND position unchanged)
            self.nav_state.steps_since_progress += 1
            logger.warning(f"No progress: {self.nav_state.steps_since_progress}/{self.stuck_threshold}")
            logger.warning(f"current map: {current_map}, current pos: {current_pos}")
            logger.warning(f"last map: {self.nav_state.last_player_map}, last pos: {self.nav_state.last_player_pos}")
        
        # Check if stuck (no progress for threshold steps)
        if self.nav_state.steps_since_progress >= self.stuck_threshold:
            logger.warning(f"Stuck for {self.nav_state.steps_since_progress} steps (threshold={self.stuck_threshold}), replanning...")
            self._reset_nav_state()
            return []
        
        # Check if path is exhausted
        if self.nav_state.path_index >= len(self.nav_state.current_path):
            logger.warning("Path exhausted but target not reached, replanning...")
            self._reset_nav_state()
            return []
        
        # Emit next batch of actions
        return self._emit_next_actions()
        
    def _emit_next_actions(self) -> List[str]:
        """
        Helper to emit next batch of actions from current path.
        
        Returns:
            List of next actions to execute (up to batch_size)
        """
        if not self.nav_state.current_path:
            return []
        
        actions_to_emit = []
        for _ in range(self.action_batch_size):
            if self.nav_state.path_index >= len(self.nav_state.current_path):
                break
            actions_to_emit.append(self.nav_state.current_path[self.nav_state.path_index])
            self.nav_state.path_index += 1
        
        if actions_to_emit:
            logger.debug(
                f"Emitting actions [{self.nav_state.path_index - len(actions_to_emit)}:"
                f"{self.nav_state.path_index}]/{len(self.nav_state.current_path)}: {actions_to_emit}"
            )
        
        return actions_to_emit
    
    def _perform_interaction(self, state_data: dict, frame: np.ndarray) -> List[str]:
        """
        Phase 3: VLM decides interaction, then reset to idle.
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            Interaction button presses
        """
        actions = self._decide_interaction_with_vlm(frame, state_data)
        self.nav_state.phase = "idle"  # reset after interaction
        return actions
    
    def _choose_target_with_vlm(
        self,
        targets: List[NavigationTarget],
        frame: np.ndarray,
        state_data: dict
    ) -> NavigationTarget:
        """
        Use VLM with structured output to choose navigation target.
        
        Args:
            targets: List of available navigation targets
            frame: Game frame
            state_data: Game state data
            
        Returns:
            Chosen NavigationTarget
        """
        # Format targets for prompt
        formatted_targets = self._format_targets_for_prompt(targets)
        
        # Build prompt using template
        prompt = TARGET_SELECTION_PROMPT.format(
            formatted_targets=formatted_targets,
            player_map_tile_x=self.player_map_tile_x,
            player_map_tile_y=self.player_map_tile_y
        )
        
        # Call VLM with structured output and automatic logging
        logger.info(f"Choosing target from {len(targets)} options")
        try:
            decision: TargetDecision = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame,
                module_name="TARGET_SELECTION",
                structured_output_model=TargetDecision
            )
            
            logger.info(f"VLM chose target index {decision.target_index}: {decision.reasoning}")
            
            # Extract chosen target using shared validation utility
            from custom_utils.navigation_targets import validate_and_select_target

            try:
                chosen_target = validate_and_select_target(decision.target_index, targets)
                logger.info(f"Selected target: {chosen_target.description}")
                return chosen_target
            except ValueError as e:
                # This shouldn't happen since validate_and_select_target has fallback logic
                logger.error(f"Target validation failed: {e}, using first target")
                return targets[0]
        except Exception as e:
            logger.error(f"VLM target selection failed: {e}, using first target")
            return targets[0]
    
    def _decide_interaction_with_vlm(
        self,
        frame: np.ndarray,
        state_data: dict
    ) -> List[str]:
        """
        Use VLM with structured output to decide interaction with reached target.
        
        Args:
            frame: Game frame
            state_data: Game state data
            
        Returns:
            List of button presses
        """
        from utils.state_formatter import format_state_summary
        
        state_summary = format_state_summary(state_data)
        
        target_description = self.nav_state.current_target.description if self.nav_state.current_target else "unknown target"
        prompt = INTERACTION_PROMPT.format(
            target_description=target_description,
            state_summary=state_summary
        )
        
        logger.info(f"Deciding interaction for target: {target_description}")
        try:
            decision: InteractionDecision = self.call_vlm_with_logging(
                prompt=prompt,
                image=frame,
                module_name="INTERACTION",
                structured_output_model=InteractionDecision
            )
            
            logger.info(f"VLM interaction reasoning: {decision.reasoning}")
            logger.info(f"VLM interaction buttons: {decision.button_names}")
            
            return decision.button_names
        except Exception as e:
            logger.error(f"VLM interaction decision failed: {e}, using default 'A'")
            return ['A']
    
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
        state_data: dict
    ) -> List[str]:
        """
        Use A* pathfinder to plan path to target.
        
        Args:
            target: Navigation target
            state_data: Game state data
            
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
        if self.movement_mode == "facing_aware" and not self.nav_state.facing_verified:
            movement_mode = "naive"
        else:
            movement_mode = self.movement_mode
        
        path = self.pathfinder.find_path(
            start_local_tile=start_local_tile,
            goal_local_tile=goal_local_tile,
            start_facing=start_facing,
            terrain=terrain,
            obstacles=None,
            movement_mode=movement_mode
        )
        
        logger.info(f"Planned path to {target.description}: {len(path)} actions (mode={movement_mode})")
        
        return path
    
    def _has_reached_target(self, state_data: dict) -> bool:
        """
        Check if player reached target (handles map transitions and distance).
        TODO: Refactor common logic with navigation_executor.py

        Args:
            state_data: Game state data

        Returns:
            True if target reached based on type-specific logic
        """
        current_target = self.nav_state.current_target
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
            # Reason: Boundaries are traversable tiles at screen edges
            if map_changed:
                logger.info(f"Boundary target '{target_description}' reached - map transitioned from '{target_source_map}' to '{current_map}'")
            elif distance == 0:
                logger.info(f"Boundary target '{target_description}' reached - player on boundary tile")
            return map_changed or distance == 0

        elif target_type == "object":
            # Objects (NPCs, items): distance-based check on same map
            # Reason: Objects don't cause map transitions, just interactions
            # TODO: refine this. NPCs are non traversable so distance should be 1
            # (and A* should handle) being adjacent
            # But there might be some edge cases of traversable objects
            reached = distance <= 1
            if reached:
                logger.info(f"Object target '{target_description}' reached at distance {distance}")
            return reached

        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def _format_targets_for_prompt(
        self,
        targets: List[NavigationTarget]
    ) -> str:
        """
        Format targets for prompt using shared utility.

        Note: Also logs each target for debugging.
        """
        from custom_utils.navigation_targets import format_targets_for_prompt

        if not targets:
            return "No targets detected."

        # Log targets for debugging
        for i, target in enumerate(targets):
            logger.info(f"Target {i}: {target.description}")

        return format_targets_for_prompt(targets)
