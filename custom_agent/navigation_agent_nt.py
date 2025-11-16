"""
Navigation agent with object detection, A* pathfinding, and VLM decision making.

"""

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field
import logging
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import deque

from custom_agent.base_agent import AgentRegistry
from custom_agent.base_langchain_agent import BaseLangChainAgent
from custom_utils.cached_object_detector_nt import CachedObjectDetector as ObjectDetector
from custom_utils.navigation_targets_nt import (
    NavigationTarget, 
    generate_navigation_targets
)
from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid, add_turn_actions
from custom_utils.map_extractor_nt import get_player_centered_grid, extract_portal_connections
from custom_utils.label_traversable import compute_simple_tile_features
from custom_utils.detectors import detect_dialogue
from custom_utils.log_to_active import (
    ensure_cache_directories, load_active_tile_index, save_active_tile_index,
    load_target_selection_counts, save_target_selection_counts,
    log_interaction, save_tile_to_cache, mark_and_save_tile,
    load_portal_connections, update_portal_connections_cache, INTERACTION_LOG_FILE
)

logger = logging.getLogger(__name__)

# Constants
NAVIGATION_CACHE_DIR = "./navigation_caches"
DIALOGUE_LOG_FILE = "./startup_cache/dialogues.json"
TARGET_SELECTION_CACHE = Path(NAVIGATION_CACHE_DIR) / "target_selection_counts.json"

# Prompt templates
TARGET_SELECTION_PROMPT = """
You are playing Pokemon Emerald. Choose where to navigate.

CURRENT OBJECTIVE:
{objective_context}

DETECTED TARGETS:
{formatted_targets}

You are the person at (7, 7) in the 15x15 grid

{dialogue_history}

Choose the target that best advances your current objective.
Prioritize targets that directly help complete the objective.

IMPORTANT: Prefer targets with lower selection counts. If a target has been selected many times without success (high number_of_times_selected), consider trying a different approach or target.
"""

INTERACTION_PROMPT = """
You are playing Pokemon Emerald and have reached a navigation target.

NAVIGATION TARGET:
{target_description}

State:
{objectives_context}

{dialogue_context}

PLAYER STATUS:
Position: ({player_x}, {player_y})
Map: {player_map}

Decide what interaction to perform. Common patterns:
- 'A' to talk to NPCs/interact with objects/open doors
- Multiple 'A' presses (e.g., ['A', 'A', 'A']) to advance through dialogue
- 'B' to cancel or go back
- Arrow keys (UP, DOWN, LEFT, RIGHT) to navigate menus

Respond with button presses to complete the interaction.
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
    
    # Step 6: Interaction trigger detection
    last_actions: List[str] = []  # Last actions taken (full list)
    last_dialogue_state: bool = False  # Previous dialogue state
    tried_interaction: bool = False  # Whether we've tried 'A' button when stuck
    tried_facing: bool = False  # Whether we've tried facing the target
    target_facing_direction: Optional[str] = None  # Direction to face target
    
    # Step 7: Failed movement detection
    movement_queue: List[str] = []  # Last 3 movement actions
    position_queue: List[tuple] = []  # Last 3 positions (x, y)
    failed_movement_count: int = 0  # Consecutive failed movements
    
    # UCB Bandit exploration override
    bandit_mode_active: bool = False  # Whether bandit override is active
    bandit_iterations: int = 0  # Number of iterations in bandit mode
    failed_target_attempts: int = 0  # Consecutive failed attempts to reach current target
    initial_distance_to_target: Optional[float] = None  # Initial Manhattan distance to target
    displacement_history: List[float] = []  # Last 5 distances to target
    bandit_start_pos: Optional[tuple] = None  # Position when bandit mode started
    bandit_start_map: Optional[str] = None  # Map when bandit mode started
    bandit_total_movement: int = 0  # Total tiles moved since bandit started
    last_map_for_bandit: Optional[str] = None  # Track map transitions for bandit
    
    # NPC-initiated dialogue detection
    last_checked_dialogue: bool = False  # Dialogue state from previous step
    npc_interrupted_navigation: bool = False  # Whether NPC interrupted navigation with dialogue
    interaction_failed: bool = False  # Whether interaction failed without dialogue (don't exclude positions)
    expected_interaction_tile: Optional[tuple] = None  # Tile we expect to interact with (set when pressing 'A')
    interaction_stuck_steps: int = 0  # Steps stuck at target without pressing 'A'
    reached_target_for_interaction: bool = False  # Whether we transitioned to interacting because we reached the target


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


@AgentRegistry.register("navigation-nt")
class NavigationAgentNT(BaseLangChainAgent):
    """Navigation agent with object detection and A* pathfinding."""
    
    def __init__(
        self, 
        backend: str = "github_models", 
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        action_batch_size: int = 100,
        stuck_threshold: int = 3,
        movement_mode: Literal["naive", "facing_aware"] = "naive",
        server_url: str = "http://localhost:8000",
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
            server_url: Server URL for pushing NPC updates
            **kwargs: Additional args passed to BaseLangChainAgent
        """
        # Check for environment-based server URL override
        env_server_host = os.getenv("AGENT_SERVER_HOST")
        env_server_port = os.getenv("AGENT_PORT")
        if env_server_host and env_server_port:
            server_url = f"http://{env_server_host}:{env_server_port}"
            logger.info(f"Using server URL from environment: {server_url}")
        elif env_server_host:
            server_url = f"http://{env_server_host}:8000"
            logger.info(f"Using server URL from environment (default port): {server_url}")
        
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
        self.server_url = server_url
        
        logger.info(f"NavigationAgentNT server_url set to: {self.server_url}")
        
        # Current state (updated each step)
        self.player_map_tile_x: Optional[int] = None
        self.player_map_tile_y: Optional[int] = None
        self.player_map: Optional[str] = None
        # Facing is inferred via probe-and-process: first action acts as probe
        self.player_facing: Optional[str] = None
        self.traversability_map: Optional[List[List[str]]] = None
        
        # NPC detection state
        self.latest_npc_detections: Optional[List[Dict]] = None
        
        # NPC updates to be returned to client
        self.pending_npc_updates: Optional[Dict] = None
        
        # Active tile cache tracking (Step 6 & 7)
        self.active_tile_index: Dict[str, Dict] = {}  # Dict of {filename: {'class': str, 'tile_pos': [x, y]}}
        ensure_cache_directories()
        self.active_tile_index = load_active_tile_index()
        
        # NOTE: seen_tiles_cache.json is currently not being used
        # Seen tiles cache for grass detection
        # self.seen_tiles_cache: Dict[str, List[tuple]] = {}  # map_name -> sorted list of (x, y)
        # self.last_grass_check_pos: Optional[tuple] = None  # Last position we checked for grass
        # self._load_seen_tiles_cache()
        
        # Target selection tracking
        self.target_selection_counts: Dict[str, int] = {}  # target_key -> selection_count
        self.target_selection_counts = load_target_selection_counts()
        
        # UCB Bandit exploration state (per map)
        self.coordinate_rewards: Dict[str, Dict[tuple, float]] = {}  # map_name -> {(x,y): avg_reward}
        self.coordinate_visits: Dict[str, Dict[tuple, int]] = {}  # map_name -> {(x,y): visit_count}
        self.global_step_counter: int = 1  # Global step counter for UCB calculation
        
        # Debug state
        self._last_frame: Optional[np.ndarray] = None
        
        logger.info(f"Initialized NavigationAgent with {backend}/{model_name}")
        logger.info(f"Config: batch_size={action_batch_size}, stuck_threshold={stuck_threshold}, movement_mode={movement_mode}")
        logger.info(f"Server URL: {server_url}")
    
    def choose_action(self, state_data: dict, frame: np.ndarray, **kwargs) -> Dict:
        """
        Main decision loop with three-phase navigation.
        
        Args:
            state_data: Game state data from emulator
            frame: Game frame as numpy array
            **kwargs: Additional arguments (including optional target_entity_filter, target_x, target_y, player_direction_result)
            
        Returns:
            Dict containing 'action' (List[str]) and optionally 'npc_updates' (Dict)
        """
        # Extract entity filter if provided (for level grinding grass targeting)
        self._target_entity_filter = kwargs.get('target_entity_filter', None)
        
        # Extract target coordinates if provided (for entity-based navigation)
        self._target_x = kwargs.get('target_x', None)
        self._target_y = kwargs.get('target_y', None)
        
        # Extract objective context if provided (for target selection)
        self._current_objective_context = kwargs.get('objective_context', None)
        
        # Extract positions to exclude from navigation (e.g., recently completed dialogues)
        self._exclude_positions = kwargs.get('exclude_positions', [])
        
        # Extract player direction result if provided by overall agent (to avoid redundant detection)
        self._player_direction_result = kwargs.get('player_direction_result', None)
        
        # Cache current frame for use in update_state
        self.curr_frame = frame.copy() if hasattr(frame, 'copy') else np.array(frame)
        
        # Update state (player + map) first
        self._update_state(state_data)
        
        # Patch traversability map with active untraversable tiles
        self._patch_traversability_with_active_tiles()
        
        # NOTE: seen_tiles_cache.json is currently not being used
        # Check for unseen grass and update caches if needed
        # self._check_unseen_grass(state_data, frame)
        
        # Step 6: Check if we pressed 'A' in previous step and need to detect result
        # This must happen BEFORE we decide next actions
        if self.nav_state.tried_interaction and self.nav_state.phase == "interacting":
            self._check_interaction_result(state_data, frame)
        
        # Check for NPC-initiated dialogue during navigation
        current_dialogue = self._detect_dialogue_state(state_data, frame)
        if (self.nav_state.phase == "navigating" and 
            current_dialogue and not self.nav_state.last_checked_dialogue and
            'A' not in self.nav_state.last_actions):
            # NPC initiated dialogue unexpectedly - interrupt navigation
            logger.warning("NPC-initiated dialogue detected during navigation - interrupting")
            self._handle_npc_initiated_dialogue(state_data, frame)
            # Transition to interacting phase to handle dialogue completion
            self.nav_state.phase = "interacting"
            self.nav_state.npc_interrupted_navigation = True
            self.nav_state.last_checked_dialogue = current_dialogue
            return {'action': []}
        
        # Update dialogue state for next check
        self.nav_state.last_checked_dialogue = current_dialogue
        
        if self.nav_state.phase in ["idle", "choosing_target"]:
            actions = self._start_new_navigation(state_data, frame)
        
        elif self.nav_state.phase == "navigating":
            actions = self._execute_navigation(state_data, frame)
        
        elif self.nav_state.phase == "interacting":
            actions = self._perform_interaction(state_data, frame)
            
            # If interaction returned empty (dialogue just completed OR interaction failed), start new navigation
            if not actions and self.nav_state.phase == "idle":
                player_data = state_data.get('player', {})
                position = player_data.get('position', {})
                player_x = position.get('x')
                player_y = position.get('y')
                
                # Check if this was an NPC-initiated dialogue interruption
                if self.nav_state.npc_interrupted_navigation:
                    logger.info("Processing post-dialogue NPC detection after interruption")
                    self._detect_and_log_npc_after_dialogue(state_data, frame)
                    self.nav_state.npc_interrupted_navigation = False
                
                # Only exclude positions if dialogue actually occurred (not if interaction just failed)
                if not self.nav_state.interaction_failed and player_x is not None and player_y is not None:
                    # Exclude current position and adjacent positions (Manhattan distance <= 1)
                    exclude_positions = []
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if abs(dx) + abs(dy) <= 1:
                                exclude_positions.append((player_x + dx, player_y + dy))
                    
                    logger.info(f"Excluding {len(exclude_positions)} positions near completed dialogue: {exclude_positions}")
                    
                    # Store in kwargs for next navigation cycle
                    self._exclude_positions = exclude_positions
                
                # Reset the flag for next interaction
                self.nav_state.interaction_failed = False
                
                # Trigger new navigation
                actions = self._start_new_navigation(state_data, frame)
        
        else:
            actions = []
        
        # Track the action we're about to take (for Step 6 & 7)
        if actions and len(actions) > 0:
            self.nav_state.last_actions = actions.copy()
            
            # Track movement queue for Step 7
            current_pos = (self.player_map_tile_x, self.player_map_tile_y)
            if current_pos[0] is not None and current_pos[1] is not None:
                self._track_movement_queue(actions[0], current_pos)
                
                # Step 7: Check for failed movement (3 consecutive attempts)
                self._check_failed_movement(frame, state_data)
            
            # Check if failed movement detection switched to interaction mode
            if self.nav_state.phase == "interacting":
                return self._perform_interaction(state_data, frame)
        
        # Prepare result dict
        result = {'action': actions}
        
        # Include NPC updates if available
        if self.pending_npc_updates is not None:
            result['npc_updates'] = self.pending_npc_updates
            # Clear pending updates after including them
            self.pending_npc_updates = None
        
        return result
    
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
        # NOTE: This uses the 15x15 visual_map grid system (player at 7,7)
        # NOT the 15x11 local frame system used by label_traversable (player at 7,5)
        map_data = state_data.get('map', {})
        
        # Always regenerate base traversability map (without NPCs) on each step
        # This ensures the grid is centered on current player position
        # Active tiles will be patched in after this via _patch_traversability_with_active_tiles()
        self.traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)],
            npc_detections=None  # No NPCs for base grid
        )
        
        # Create debug overlay and update traversability map with NPC detections
        # This prints before/after and returns the updated grid
        # if hasattr(self, 'curr_frame') and self.curr_frame is not None:
        #     updated_map = self._create_debug_traversability_overlay(
        #         self.curr_frame,
        #         self.traversability_map,
        #         navigation_targets=None,  # No targets yet in update_state
        #         state_data=state_data,
        #         filename_suffix="_update"  # Use _update suffix for update_state calls
        #     )
            
        #     # Save updated map with NPCs
        #     if updated_map is not None:
        #         self.traversability_map = updated_map
        
        # Store frame for next debug overlay (if needed)
        if hasattr(self, 'curr_frame') and self.curr_frame is not None:
            self._last_frame = self.curr_frame.copy() if hasattr(self.curr_frame, 'copy') else np.array(self.curr_frame)
        
        # Store NPC updates for client to push to server
        # Note: This replaces direct posting - client will handle the HTTP request
        if getattr(self, 'latest_npc_detections', None):
            logger.debug("Storing NPC updates for client to push to server")
            try:
                self.pending_npc_updates = self._prepare_npc_updates_payload()
            except Exception as e:
                # Fail gracefully - NPC updates are optional for visualization only
                logger.debug(f"Could not prepare NPC updates (optional): {e}")
                self.pending_npc_updates = None
        else:
            logger.debug("No NPC detections - clearing pending updates")
            self.pending_npc_updates = None
        
        # Extract and cache portal connections from visual_map when map changes
        visual_map = state_data.get('visual_map', '')
        if visual_map and self.player_map and self.player_map != self.nav_state.last_map_for_bandit:
            portal_connections = extract_portal_connections(visual_map)
            logger.info(f"Extracted {len(portal_connections)} portal connections from visual_map for {self.player_map}: {portal_connections}")
            if portal_connections:
                update_portal_connections_cache(self.player_map, portal_connections)
                logger.debug(f"Updated portal connections cache for {self.player_map}: {len(portal_connections)} connections")
            else:
                logger.debug(f"No portal connections found in visual_map for {self.player_map}")
            self.nav_state.last_map_for_bandit = self.player_map
    
    def _start_new_navigation(self, state_data: dict, frame: np.ndarray) -> List[str]:
        """
        Phase 1: Detect objects, generate targets, choose with VLM, plan naive path.
        
        This function handles the initial navigation setup:
        1. Extracts and caches portal connections from visual_map
        2. Detects objects using object_detector.detect_objects (unless entity target provided)
        3. Generates navigation targets including portals, boundaries, stairs, doors
        4. Chooses target using VLM (unless bandit mode is active)
        5. Plans A* path to chosen target
        
        Bandit mode activation conditions:
        - 5+ consecutive failed attempts to reach a target
        - Displacement to target has not improved by at least 30% over last 5 steps
        - Automatically deactivates after 10 iterations or 10+ tiles moved
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            First action (used as probe to determine facing)
        """
        logger.info("Starting new navigation phase")
        
        # Load portal connections from cache
        portal_connections = load_portal_connections()
        if portal_connections:
            logger.debug(f"Loaded {len(portal_connections)} portal connections for {self.player_map}")
        else:
            logger.debug(f"No portal connections cached for {self.player_map}")
        
        # Check for entity type filter in kwargs (passed from overall agent)
        target_entity_filter = getattr(self, '_target_entity_filter', None)
        
        # Check if specific target coordinates were provided (from entity database)
        target_x = getattr(self, '_target_x', None)
        target_y = getattr(self, '_target_y', None)
        
        # If entity target is provided, skip object detection entirely
        if (target_x is not None and target_y is not None) or self.nav_state.bandit_mode_active:
            logger.info(f"Entity database target provided: ({target_x}, {target_y}) - skipping object detection")
            detected_objects = []
        else:
            # Step 1: Detect objects (skip for MOVING_VAN interior)
            if self.player_map == "MOVING_VAN":
                logger.info("MOVING_VAN interior - skipping object detection")
                detected_objects = []
            else:
                logger.info("Detecting objects in frame")
                # object_detector.detect_objects: Analyzes the game frame to identify interactive objects
                # such as NPCs, doors, stairs, items, and grass patches using computer vision
                detected_objects = self.object_detector.detect_objects(frame)
                logger.info(f"Detected {len(detected_objects)} objects")
            
            # Filter by entity type if specified
            if target_entity_filter is not None:
                detected_objects = [obj for obj in detected_objects 
                                  if obj.get('entity_type') in target_entity_filter]
                logger.info(f"After entity type filtering: {len(detected_objects)} objects")

        # Step 1.5: Detect NPCs via exact tile matching
        logger.info("Running NPC tile detection")
        if self.player_map == "LITTLEROOT TOWN":
            logger.info("Skipping NPC tile detection for LITTLEROOT TOWN")
            npc_detected_objects = []
            self.latest_npc_detections = None
        else:
            try:
                npc_detected_objects = self.object_detector.detect_exact_tile_matches(frame)
                logger.info(f"🔍 Detected {len(npc_detected_objects)} NPC tiles in current frame")
                
                # Extend detected_objects with NPC detections
                detected_objects.extend(npc_detected_objects)
                
                # Convert DetectedObject format to dict format expected by get_player_centered_grid
                npc_detections = []
                for npc_obj in npc_detected_objects:
                    if npc_obj.source == "uniquetiles_match":
                        center_x, center_y = npc_obj.center_pixel
                        npc_detections.append({
                            'center_x': center_x,
                            'center_y': center_y,
                            'score': npc_obj.confidence,
                            'cluster_id': npc_obj.name.replace('cluster_', '') if npc_obj.name.startswith('cluster_') else npc_obj.name
                        })
                
                # Store NPC detections for later overlay on visual map
                self.latest_npc_detections = npc_detections
            except Exception as e:
                logger.error(f"Failed to detect NPC tiles: {e}")
                import traceback
                traceback.print_exc()
                self.latest_npc_detections = None

        # Step 2: Generate navigation targets
        player_map_tile_pos = (self.player_map_tile_x, self.player_map_tile_y)
        
        # Check if specific target coordinates were provided (from entity database)
        target_x = getattr(self, '_target_x', None)
        target_y = getattr(self, '_target_y', None)
        
        if target_x is not None and target_y is not None:
            # Create a single target from provided coordinates
            logger.info(f"Using entity database target: ({target_x}, {target_y})")
            from custom_utils.navigation_targets import NavigationTarget
            
            # Calculate local tile position from map tile position
            # Player is at (7, 7) in local grid, so offset from player position
            player_local_x = 7
            player_local_y = 7
            local_x = player_local_x + (target_x - player_map_tile_pos[0])
            local_y = player_local_y + (target_y - player_map_tile_pos[1])
            
            targets = [NavigationTarget(
                id=f"entity_target_{target_x}_{target_y}",
                type="object",  # Treat entity targets as objects
                map_tile_position=(target_x, target_y),
                local_tile_position=(local_x, local_y),
                description=f"Entity target at ({target_x}, {target_y})",
                entity_type="entity_database",
                source_map_location=self.player_map
            )]
            
            # Add portal targets even for entity navigation
            for m, connection in enumerate(portal_connections):
                from_pos = connection['from_pos']
                to_map = connection['to_map']
                
                # Convert map position to local tile position
                local_tile_x = 7 + (from_pos[0] - player_map_tile_pos[0])
                local_tile_y = 7 + (from_pos[1] - player_map_tile_pos[1])
                
                # Only include if within the 15x15 grid
                if 0 <= local_tile_x < 15 and 0 <= local_tile_y < 15:
                    portal_target = NavigationTarget(
                        id=f"portal_{m}",
                        type="door",
                        map_tile_position=from_pos,
                        local_tile_position=(local_tile_x, local_tile_y),
                        description=f"Door at ({from_pos[0]}, {from_pos[1]}) in {self.player_map} to go to {to_map}",
                        priority=0.0,
                        entity_type="portal",
                        detected_object=None,
                        source_map_location=self.player_map,
                        tile_size=(1, 1)
                    )
                    targets.append(portal_target)
                    logger.debug(f"Added portal target to {to_map} at {from_pos}")
            
            logger.info(f"Created navigation target from entity database coordinates")
        else:
            # Normal target generation from detected objects
            logger.info("Generating navigation targets")
            include_grass = (target_entity_filter == 'grass')
            targets = generate_navigation_targets(
                detected_objects=detected_objects,
                traversability_map=self.traversability_map,
                player_map_tile_pos=player_map_tile_pos,
                player_map_location=self.player_map,
                include_grass_targets=include_grass,
                portal_connections=portal_connections
            )
            logger.info(f"Generated {len(targets)} navigation targets")
            
            # Filter out NPCs that have already been interacted with
            completed_npcs = self._get_completed_npc_interactions()
            if completed_npcs:
                # Create a set of (map, x, y) tuples for quick lookup
                completed_positions = {
                    (npc['map_location'], npc['tile_pos'][0], npc['tile_pos'][1])
                    for npc in completed_npcs
                    if len(npc.get('tile_pos', [])) == 2
                }
                
                # Filter out targets that match completed NPC positions
                original_count = len(targets)
                targets = [
                    target for target in targets
                    if (target.source_map_location, target.map_tile_position[0], target.map_tile_position[1]) 
                    not in completed_positions
                ]
                
                filtered_count = original_count - len(targets)
                if filtered_count > 0:
                    logger.info(f"Filtered out {filtered_count} already-completed NPC targets")
        
        # Debug: Log target types before overlap resolution
        target_types_before = {}
        for target in targets:
            target_types_before[target.type] = target_types_before.get(target.type, 0) + 1
        logger.info(f"Target types before overlap resolution: {target_types_before}")
        
        # Resolve overlapping targets
        targets = self._resolve_target_overlaps(targets)
        logger.info(f"After overlap resolution: {len(targets)} navigation targets")
        
        # Filter out excluded positions (e.g., recently completed dialogues)
        exclude_positions = getattr(self, '_exclude_positions', [])
        if exclude_positions:
            original_count = len(targets)
            targets = [
                target for target in targets
                if target.map_tile_position not in exclude_positions
            ]
            filtered_count = original_count - len(targets)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} excluded position targets")
        
        # Filter for grass targets if entity filter specifies grass
        if target_entity_filter == 'grass':
            original_count = len(targets)
            grass_targets = [target for target in targets if target.type == "grass"]
            if len(grass_targets) > 0:
                targets = grass_targets
                filtered_count = original_count - len(targets)
                logger.info(f"Filtered for grass targets: kept {len(targets)} grass targets (filtered out {filtered_count})")
            else:
                # No grass targets found, keep all targets as fallback
                logger.info(f"No grass targets found, using all {len(targets)} targets")
        
        # Filter for clock targets if objective mentions clock
        objective_context = getattr(self, '_current_objective_context', None) or 'No specific objective - explore and interact'
        if 'clock' in objective_context.lower():
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
        objective_context = getattr(self, '_current_objective_context', None) or 'No specific objective - explore and interact'
        if 'clock' in objective_context.lower() and self.player_map == "LITTLEROOT TOWN BRENDANS HOUSE 2F":
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
                        
                        from custom_utils.navigation_targets import NavigationTarget
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
        
        # # Debug: Create side-by-side traversability overlay with navigation targets
        # # This is just for visualization - we don't save the updated map here
        # # (it was already saved in _update_state)
        self._create_debug_traversability_overlay(
            frame, 
            self.traversability_map,
            navigation_targets=targets,
            state_data=state_data,
            filename_suffix="_combined"  # Use _combined suffix for start_new_navigation calls
        )
        
        if not targets:
            logger.warning("No targets found, returning empty action")
            self.nav_state.phase = "idle"
            return []
        
        # Step 3: Choose target with VLM
        chosen_target = self._choose_target_with_vlm(targets, frame, state_data)
        
        # Debug: Create another overlay showing the chosen target
        self._create_debug_traversability_overlay(
            frame, 
            self.traversability_map,
            navigation_targets=targets,
            state_data=state_data,
            filename_suffix="_chosen",
            chosen_target=chosen_target  # Pass chosen target for highlighting
        )
        
        # Debug: Export target information to JSON
        self._export_targets_json(targets, chosen_target, filename_suffix="_chosen")
        
        # Step 4: Plan path to target
        logger.info(f"Planning path to target: {chosen_target.description}")
        path = self._plan_path_to_target(chosen_target, state_data)
        
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
        self.nav_state.last_map_for_bandit = None  # Reset map tracking
        self.nav_state.facing_verified = False  # Reset for next navigation
        self.nav_state.tried_interaction = False  # Reset interaction flag
        self.nav_state.tried_facing = False  # Reset facing flag
        self.nav_state.target_facing_direction = None  # Reset facing direction
        self.nav_state.expected_interaction_tile = None  # Reset expected interaction tile
        self.nav_state.interaction_stuck_steps = 0  # Reset stuck counter
        self.nav_state.reached_target_for_interaction = False  # Reset reached target flag
        self.nav_state.last_actions = []  # Reset last actions
        self.nav_state.failed_movement_count = 0  # Reset failed movement count
        
        # Reset displacement tracking
        self.nav_state.failed_target_attempts = 0
        self.nav_state.initial_distance_to_target = None
        self.nav_state.displacement_history = []
        
        # Clean up bandit tracking attributes
        if hasattr(self, '_last_bandit_pos'):
            delattr(self, '_last_bandit_pos')
        if hasattr(self, '_last_bandit_map'):
            delattr(self, '_last_bandit_map')
    
    def _get_attempted_interaction_tile(self, player_x: int, player_y: int) -> Optional[tuple]:
        """
        Unified function to determine the tile we attempted to interact with or move to.
        
        Combines logic from various interaction and movement detection functions.
        Priority order:
        1. Expected interaction tile (set when pressing 'A')
        2. Current target if adjacent
        3. From last movement action in queue
        4. From last action in current path
        5. From detected facing direction
        
        Args:
            player_x: Current player X position
            player_y: Current player Y position
            
        Returns:
            (x, y) tuple of attempted tile, or None if cannot determine
        """
        # 1. First check if we stored an expected interaction tile (highest priority)
        if self.nav_state.expected_interaction_tile is not None:
            return self.nav_state.expected_interaction_tile
        
        # 2. Check if we have a current target and are adjacent to it
        if self.nav_state.current_target:
            target_pos = self.nav_state.current_target.map_tile_position
            distance = abs(player_x - target_pos[0]) + abs(player_y - target_pos[1])
            if distance == 1:
                return target_pos
        
        # 3. Try to determine from last movement action in queue (most recent movement)
        if self.nav_state.movement_queue:
            # Get most recent movement action
            for action in reversed(self.nav_state.movement_queue):
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
        if self.nav_state.current_path and self.nav_state.path_index > 0:
            last_action_index = self.nav_state.path_index - 1
            if last_action_index < len(self.nav_state.current_path):
                last_action = self.nav_state.current_path[last_action_index]
                
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
        
        # 5. Fallback: use detected facing direction
        if self.player_facing and player_x is not None and player_y is not None:
            facing_offset = {
                'North': (0, -1),
                'South': (0, 1),
                'East': (1, 0),
                'West': (-1, 0)
            }
            offset = facing_offset.get(self.player_facing, (0, 0))
            next_tile = (player_x + offset[0], player_y + offset[1])
            return next_tile
        
        return None

    def _execute_navigation(self, state_data: dict, frame: np.ndarray) -> List[str]:
        """
        Phase 2: Execute path step-by-step, check for arrival or stuck.
        
        Monitors progress and activates bandit mode if stuck:
        - Tracks displacement to target over last 5 steps
        - Activates bandit exploration if displacement hasn't improved by 30%
        - Bandit mode uses UCB algorithm to explore high-reward coordinates
        - Deactivates after 10 iterations or significant movement (>10 tiles)
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            Next action(s) in path
        """
        # Check if reached target
        if self._has_reached_target(state_data):
            logger.info("Reached target, transitioning to interaction phase")
            
            # Reward the target coordinate for successful reach
            if self.nav_state.bandit_mode_active and self.nav_state.current_target:
                target_coord = self.nav_state.current_target.map_tile_position
                self._update_coordinate_reward(current_map, target_coord, reward=2.0)  # Higher reward for reaching target
                logger.debug(f"Bandit mode: rewarded target coordinate {target_coord} for being reached")
            
            self.nav_state.phase = "interacting"
            self.nav_state.reached_target_for_interaction = True
            return []
        
        # Track player state (map + position) to detect progress
        current_pos = (self.player_map_tile_x, self.player_map_tile_y)
        current_map = self.player_map

        # Initialize tracking on first iteration
        if self.nav_state.last_player_pos is None or self.nav_state.last_player_map is None:
            self.nav_state.last_player_pos = current_pos
            self.nav_state.last_player_map = current_map
            logger.debug(f"Initialized navigation tracking: map={current_map}, pos={current_pos}")

        position_changed = current_map != self.nav_state.last_player_map or current_pos != self.nav_state.last_player_pos
        
        # Track displacement to target for bandit override detection
        if self.nav_state.current_target and current_pos[0] is not None and current_pos[1] is not None:
            target_pos = self.nav_state.current_target.map_tile_position
            current_distance = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])
            
            # Initialize distance tracking on first measurement
            if self.nav_state.initial_distance_to_target is None:
                self.nav_state.initial_distance_to_target = current_distance
                self.nav_state.displacement_history = [current_distance]
            else:
                # Add to displacement history (keep last 5)
                self.nav_state.displacement_history.append(current_distance)
                if len(self.nav_state.displacement_history) > 5:
                    self.nav_state.displacement_history.pop(0)

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
            self.nav_state.tried_interaction = False  # Reset interaction flag on progress
            self.nav_state.failed_movement_count = 0  # Reset failed movement count on progress
            self.nav_state.last_player_pos = current_pos
            self.nav_state.last_player_map = current_map
            
            # Update coordinate reward (positive for progress)
            if self.nav_state.bandit_mode_active:
                self._update_coordinate_reward(current_map, current_pos, reward=1.0)
                logger.debug(f"📈 Bandit mode: rewarded coordinate {current_pos} for progress")
            
            # Check if displacement improved sufficiently (>30% reduction)
            if (self.nav_state.initial_distance_to_target is not None and 
                len(self.nav_state.displacement_history) >= 2):
                initial_dist = self.nav_state.displacement_history[0]
                current_dist = self.nav_state.displacement_history[-1]
                if initial_dist > 0:
                    improvement = (initial_dist - current_dist) / initial_dist
                    if improvement < 0.3:  # Less than 30% improvement
                        logger.warning(f"Insufficient progress: only {improvement*100:.1f}% improvement toward target")
                else:
                    # Already at target or very close
                    pass
        else:
            # No progress (map AND position unchanged)
            self.nav_state.steps_since_progress += 1
            logger.warning(f"No progress: {self.nav_state.steps_since_progress}/{self.stuck_threshold}")
            logger.warning(f"current map: {current_map}, current pos: {current_pos}")
            logger.warning(f"last map: {self.nav_state.last_player_map}, last pos: {self.nav_state.last_player_pos}")
            
            # Only increment failed attempts if we're genuinely stuck (at threshold)
            # Don't increment during normal path execution
            if self.nav_state.steps_since_progress >= self.stuck_threshold:
                self.nav_state.failed_target_attempts += 1
                logger.warning(f"Stuck at threshold, failed_target_attempts: {self.nav_state.failed_target_attempts}")
        
        # Check if stuck (no progress for threshold steps)
        if self.nav_state.steps_since_progress >= self.stuck_threshold:
            # Penalize the target coordinate for being unreachable
            if self.nav_state.bandit_mode_active and self.nav_state.current_target:
                target_coord = self.nav_state.current_target.map_tile_position
                self._update_coordinate_reward(current_map, target_coord, reward=-1.0)  # Penalize unreachable targets
                logger.debug(f"Bandit mode: penalized target coordinate {target_coord} for being unreachable")
            
            # Transition to interacting phase to handle stuck situation
            logger.warning(f"Stuck for {self.nav_state.steps_since_progress} steps - transitioning to interacting phase")
            self.nav_state.phase = "interacting"
            self.nav_state.reached_target_for_interaction = False
            return self._perform_interaction(state_data, frame)
        
        # Check if path is exhausted
        if self.nav_state.path_index >= len(self.nav_state.current_path):
            # Penalize the target coordinate for path exhaustion
            if self.nav_state.bandit_mode_active and self.nav_state.current_target:
                target_coord = self.nav_state.current_target.map_tile_position
                self._update_coordinate_reward(current_map, target_coord, reward=-0.5)  # Penalize for path exhaustion
                logger.debug(f"Bandit mode: penalized target coordinate {target_coord} for path exhaustion")
            
            # Transition to interacting phase to handle path exhaustion
            logger.warning("Path exhausted but target not reached - transitioning to interacting phase")
            self.nav_state.phase = "interacting"
            self.nav_state.reached_target_for_interaction = False
            return self._perform_interaction(state_data, frame)
        
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
        Phase 3: Handle interaction with reached target OR stuck situation.
        
        For reached targets: Face the target, wait to see if stuck, then press 'A' if stuck.
        For stuck situations: Press 'A' immediately.
        
        Args:
            state_data: Game state data
            frame: Game frame
            
        Returns:
            Interaction button presses or empty list to trigger new navigation
        """
        from custom_utils.detectors import detect_dialogue
        
        # Get player position
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        player_x = position.get('x')
        player_y = position.get('y')
        player_map = player_data.get('location')
        
        # Handle stuck situations (path exhausted or movement stuck) - press 'A' immediately
        if not self.nav_state.reached_target_for_interaction:
            # Check if dialogue is currently active
            dialogue_active = detect_dialogue(frame, threshold=0.45)
            
            # If not already tried interaction, press 'A'
            if not self.nav_state.tried_interaction:
                # Determine which tile we're trying to interact with
                next_tile = self._get_attempted_interaction_tile(player_x, player_y)
                
                if next_tile:
                    # Save tile to cache first (will be confirmed on next step)
                    logger.info(f"Saving tile {next_tile} to cache before interaction")
                    # Save with temporary class - will be updated in _check_interaction_result
                    mark_and_save_tile(
                        next_tile, frame, 'interacting', self.active_tile_index, player_map, 
                        allow_upgrade=True, player_map_tile_x=self.player_map_tile_x, 
                        player_map_tile_y=self.player_map_tile_y
                    )
                
                logger.info(f"Pressing 'A' to interact (stuck situation, target tile: {next_tile})")
                self.nav_state.tried_interaction = True
                self.nav_state.expected_interaction_tile = next_tile
                return ['A']
            
            # If no active dialogue, check if dialogue just completed
            if not dialogue_active:
                recent_completion = self._check_recent_dialogue_completion(player_x, player_y, player_map)
                
                if recent_completion:
                    logger.info(f"Detected recent dialogue completion at ({player_x}, {player_y}), bypassing VLM interaction")
                    
                    # Log completed dialogue to interactions.json
                    self._log_completed_dialogue_interaction(
                        npc_pos=(player_x, player_y),
                        map_location=player_map,
                        dialogue_info=recent_completion
                    )
                    
                    # Reset to idle and return empty to trigger new navigation
                    # The calling code will handle excluding nearby positions
                    self.nav_state.phase = "idle"
                    return []
            
            # Normal VLM interaction for dialogue
            actions = self._decide_interaction_with_vlm(frame, state_data)
            self.nav_state.phase = "idle"  # reset after interaction
            return actions
        
        # Handle reached target - don't automatically press 'A', wait for stuck
        else:
            # Check if we need to face the target
            if not self.nav_state.facing_verified:
                facing_action = self._get_facing_action_for_target()
                if facing_action:
                    logger.info(f"🎯 Facing target: {facing_action}")
                    return [facing_action]
                else:
                    self.nav_state.facing_verified = True
            
            # Facing verified, check if stuck at this position
            current_pos = (self.player_map_tile_x, self.player_map_tile_y)
            if self.nav_state.last_player_pos == current_pos:
                self.nav_state.interaction_stuck_steps += 1
            else:
                self.nav_state.interaction_stuck_steps = 0
            
            self.nav_state.last_player_pos = current_pos
            
            # If stuck for 3+ steps, press 'A'
            if self.nav_state.interaction_stuck_steps >= 3:
                logger.info(f"🎯 Stuck at target for {self.nav_state.interaction_stuck_steps} steps - pressing 'A' to interact")
                
                self.nav_state.tried_interaction = True
                self.nav_state.expected_interaction_tile = (self.player_map_tile_x, self.player_map_tile_y)
                
                # Save tile to cache with 'interacting' class before pressing A
                tile_filename = f"{self.player_map}_{self.player_map_tile_x}_{self.player_map_tile_y}.png"
                self.active_tile_index[tile_filename] = {
                    'class': 'interacting',
                    'tile_pos': [self.player_map_tile_x, self.player_map_tile_y]
                }
                save_active_tile_index(self.active_tile_index)
                
                return ['A']
            else:
                # Not stuck yet, wait
                logger.debug(f"🎯 At target but not stuck yet ({self.nav_state.interaction_stuck_steps}/3 steps) - waiting")
                return []
    
    def _should_activate_bandit_override(self) -> bool:
        """
        Check if bandit exploration override should be activated.
        
        Activates when:
        - 5+ failed attempts to reach target
        - Displacement has not improved by at least 30%
        
        Returns:
            True if bandit mode should be activated
        """
        # # DISABLED: Bandit mode disabled by setting impossible threshold
        # return False
        
        if self.nav_state.failed_target_attempts < 5:
            return False
        
        if (self.nav_state.initial_distance_to_target is None or 
            len(self.nav_state.displacement_history) < 5):
            return False
        
        initial_dist = self.nav_state.displacement_history[0]
        current_dist = self.nav_state.displacement_history[-1]
        
        if initial_dist == 0:
            return False  # Already at target
        
        improvement = (initial_dist - current_dist) / initial_dist
        
        if improvement < 0.3:
            logger.warning(f"Activating bandit override: only {improvement*100:.1f}% improvement after 5 attempts")
            return True
        
        return False
    
    def _should_deactivate_bandit_mode(self, current_pos: tuple, current_map: str) -> bool:
        """
        Check if bandit mode should be deactivated.
        
        Deactivates when:
        - 10+ bandit iterations completed
        - Total movement from bandit start > 10 tiles (across map transitions)
        
        Args:
            current_pos: Current player position (x, y)
            current_map: Current map name
        
        Returns:
            True if should resume VLM control
        """
        if self.nav_state.bandit_iterations >= 10:
            logger.info("Deactivating bandit mode: 10 iterations completed")
            return True
        
        # Track cumulative movement across map transitions
        if self.nav_state.bandit_start_pos is not None and self.nav_state.bandit_start_map is not None:
            # Update total movement if position or map changed
            if hasattr(self, '_last_bandit_pos') and hasattr(self, '_last_bandit_map'):
                if current_map == self._last_bandit_map:
                    # Same map - add Manhattan distance
                    movement = abs(current_pos[0] - self._last_bandit_pos[0]) + \
                              abs(current_pos[1] - self._last_bandit_pos[1])
                    self.nav_state.bandit_total_movement += movement
                elif current_map != self._last_bandit_map:
                    # Map changed - count as 1 tile movement
                    self.nav_state.bandit_total_movement += 1
                    logger.debug(f"Bandit map transition: {self._last_bandit_map} -> {current_map}, +1 movement")
            
            # Store current position for next iteration
            self._last_bandit_pos = current_pos
            self._last_bandit_map = current_map
            
            if self.nav_state.bandit_total_movement > 10:
                logger.info(f"Deactivating bandit mode: moved {self.nav_state.bandit_total_movement} tiles from start")
                return True
        else:
            # Initialize tracking
            self._last_bandit_pos = current_pos
            self._last_bandit_map = current_map
        
        return False
    
    def _initialize_coordinate_reward(self, map_name: str, coord: tuple, default_reward: float = 0.0) -> float:
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
    
    def _update_coordinate_reward(self, map_name: str, coord: tuple, reward: float) -> None:
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
    
    def _compute_ucb(self, map_name: str, coord: tuple) -> float:
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
    
    def _select_target_with_ucb_bandit(
        self,
        targets: List[NavigationTarget],
        current_map: str,
        current_pos: tuple
    ) -> NavigationTarget:
        """
        Select a target using UCB bandit strategy with weighted preferences.
        
        Priority weighting:
        - In houses (map contains '{int}F'): stairs/doors get 2x weight
        - Otherwise: boundaries + stairs/doors equal weight
        - If map transition occurred: halve probability of opposite direction
        
        Args:
            targets: List of available navigation targets
            current_map: Current map name
            
        Returns:
            Selected target
        """
        import re
        
        # Check if we're in a house (map name contains pattern like '1F', '2F', etc.)
        in_house = bool(re.search(r'\d+F', current_map)) if current_map else False
        
        # Detect map transition
        map_changed = (self.nav_state.last_map_for_bandit is not None and 
                      current_map != self.nav_state.last_map_for_bandit)
        
        # Determine which direction we came from (if map changed)
        came_from_direction = None
        if map_changed and self.nav_state.current_target:
            # Get the direction of the last target we selected
            target_desc = self.nav_state.current_target.description.lower()
            if 'north' in target_desc:
                came_from_direction = 'south'  # We came from south if we went north
            elif 'south' in target_desc:
                came_from_direction = 'north'
            elif 'west' in target_desc:
                came_from_direction = 'east'
            elif 'east' in target_desc:
                came_from_direction = 'west'
            
            logger.info(f"Map transition detected: {self.nav_state.last_map_for_bandit} -> {current_map}, came from {came_from_direction}")
        
        # Update last map
        self.nav_state.last_map_for_bandit = current_map
        
        # Ensure coordinate dictionaries are initialized for this map
        if current_map not in self.coordinate_rewards:
            self.coordinate_rewards[current_map] = {}
        if current_map not in self.coordinate_visits:
            self.coordinate_visits[current_map] = {}
        
        # Filter for navigation targets (boundaries, stairs, doors, clocks)
        # Exclude generic objects which are for interaction, not exploration
        nav_targets = [t for t in targets if t.type in ["boundary", "stairs", "door", "clock"]]
        
        if not nav_targets:
            logger.warning("No navigation targets for bandit, falling back to all targets")
            nav_targets = targets
        
        # Filter and weight targets
        weighted_targets = []
        for target in nav_targets:
            # Initialize coordinate reward if not already done
            coord = target.map_tile_position
            self._initialize_coordinate_reward(current_map, coord)
            
            # Base UCB score
            ucb = self._compute_ucb(current_map, coord)
            
            # Apply type-based weighting
            if in_house and target.type in ["stairs", "door"]:
                weight = 2.0
                logger.debug(f"{target.type.capitalize()} {target.description}: UCB={ucb:.3f}, weight=2.0 (in house)")
            else:
                weight = 1.0
                logger.debug(f"{target.type.capitalize()} {target.description}: UCB={ucb:.3f}, weight=1.0")
            
            # Apply directional preferences - prefer north
            target_desc = target.description.lower()
            if 'north' in target_desc:
                weight *= 2.0  # Double weight for north targets
                logger.debug(f"  Doubling weight for north direction: weight={weight}")
            
            # Apply directional penalty if we just came from opposite direction
            if came_from_direction and target.type == "boundary":
                target_direction = None
                if 'north' in target_desc:
                    target_direction = 'north'
                elif 'south' in target_desc:
                    target_direction = 'south'
                elif 'west' in target_desc:
                    target_direction = 'west'
                elif 'east' in target_desc:
                    target_direction = 'east'
                
                # Penalize opposite direction (halve weight)
                if target_direction == came_from_direction:
                    weight *= 0.5
                    logger.debug(f"  Halving weight for opposite direction {target_direction}: weight={weight}")
            
            # Apply distance-based weighting - prefer targets at least 10 tiles away
            if current_pos and current_pos[0] is not None and current_pos[1] is not None:
                distance = abs(coord[0] - current_pos[0]) + abs(coord[1] - current_pos[1])
                if distance >= 10:
                    weight *= 1.5  # Boost distant targets
                    logger.debug(f"  Boosting weight for distant target ({distance} tiles): weight={weight}")
                elif distance < 5:
                    weight *= 0.7  # Penalize very close targets
                    logger.debug(f"  Penalizing weight for close target ({distance} tiles): weight={weight}")
            
            weighted_score = ucb * weight
            weighted_targets.append((weighted_score, target))
        
        if not weighted_targets:
            logger.warning("No targets available for UCB selection, using first target")
            return targets[0] if targets else None
        
        # Select target with maximum weighted score
        _, selected_target = max(weighted_targets, key=lambda x: x[0])
        
        logger.info(f"UCB bandit selected: {selected_target.description} at {selected_target.map_tile_position}")
        return selected_target
    
    def _choose_target_with_vlm(
        self,
        targets: List[NavigationTarget],
        frame: np.ndarray,
        state_data: dict
    ) -> NavigationTarget:
        """
        Use VLM with structured output to choose navigation target.
        
        Can override to UCB bandit exploration when VLM target selection is stuck.
        
        Args:
            targets: List of available navigation targets
            frame: Game frame
            state_data: Game state data
            
        Returns:
            Chosen NavigationTarget
        """
        current_pos = (self.player_map_tile_x, self.player_map_tile_y)
        current_map = self.player_map
        
        # Check if bandit mode should be activated
        if not self.nav_state.bandit_mode_active and self._should_activate_bandit_override():
            logger.warning("🎰 Activating UCB bandit exploration override")
            self.nav_state.bandit_mode_active = True
            self.nav_state.bandit_iterations = 0
            self.nav_state.bandit_start_pos = current_pos
            self.nav_state.bandit_start_map = current_map
            self.nav_state.bandit_total_movement = 0
        # Force bandit mode to take over
        # self.nav_state.bandit_mode_active = True
        # self.nav_state.bandit_iterations = 0
        # self.nav_state.bandit_start_pos = current_pos
        # self.nav_state.bandit_start_map = current_map
        # self.nav_state.bandit_total_movement = 0

        # Check if bandit mode should be deactivated
        if self.nav_state.bandit_mode_active and self._should_deactivate_bandit_mode(current_pos, current_map):
            logger.info("🎯 Resuming VLM target selection")
            self.nav_state.bandit_mode_active = False
            self.nav_state.bandit_iterations = 0
            self.nav_state.bandit_start_pos = None
            self.nav_state.bandit_start_map = None
            self.nav_state.bandit_total_movement = 0
        
        # Use UCB bandit if in bandit mode
        if self.nav_state.bandit_mode_active:
            self.nav_state.bandit_iterations += 1
            logger.info(f"🎰 UCB Bandit iteration {self.nav_state.bandit_iterations}")
            
            chosen_target = self._select_target_with_ucb_bandit(targets, current_map, current_pos)
            
            # Update coordinate reward based on whether we reached the target
            # (This will be updated in next step based on progress)
            if chosen_target:
                coord = chosen_target.map_tile_position
                # Initialize visit tracking and increment visit count
                if current_map not in self.coordinate_visits:
                    self.coordinate_visits[current_map] = {}
                if coord not in self.coordinate_visits[current_map]:
                    self.coordinate_visits[current_map][coord] = 0
                
                # Increment visit count for selected target
                self.coordinate_visits[current_map][coord] += 1
                self.global_step_counter += 1
                
                logger.debug(f"Bandit selected target at {coord}, visit count now {self.coordinate_visits[current_map][coord]}")
            
            return chosen_target
        
        # Normal VLM target selection
        # Format targets for prompt
        formatted_targets = self._format_targets_for_prompt(targets)
        
        # Get completed NPC dialogues for dialogue history
        completed_npcs = self._get_completed_npc_dialogues()
        
        # Format dialogue history for prompt
        dialogue_history = ""
        if completed_npcs:
            npc_locations = []
            for npc in completed_npcs:
                pos = npc.get('npc_tile_position', {})
                map_name = npc.get('player_map', 'unknown')
                npc_locations.append(f"({pos.get('x', '?')}, {pos.get('y', '?')}) on {map_name}")
            
            dialogue_history = f'''You have visited and spoken to the NPCs at the following locations before:
{', '.join(npc_locations)}

Avoid selecting these NPCs as the target unless your goal requires you to talk to them.'''
        
        print("DEBUG DIALOGUE HISTORY:", dialogue_history)
        # Get objective context from kwargs if provided by overall agent
        objective_context = getattr(self, '_current_objective_context', 'No specific objective - explore and interact')
        
        # Build prompt using template
        prompt = TARGET_SELECTION_PROMPT.format(
            objective_context=objective_context,
            formatted_targets=formatted_targets,
            player_map_tile_x=self.player_map_tile_x,
            player_map_tile_y=self.player_map_tile_y,
            dialogue_history=dialogue_history
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
                
                # Increment selection count for this target
                target_key = f"{chosen_target.description}_{chosen_target.map_tile_position[0]}_{chosen_target.map_tile_position[1]}_{self.player_map}"
                self.target_selection_counts[target_key] = self.target_selection_counts.get(target_key, 0) + 1
                save_target_selection_counts(self.target_selection_counts)
                logger.info(f"Target {target_key} selected {self.target_selection_counts[target_key]} times")
                
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
        from custom_utils.detectors import detect_dialogue
        
        # Get target description
        target_description = self.nav_state.current_target.description if self.nav_state.current_target else "unknown target"
        
        # Get player position and map
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        player_x = position.get('x', '?')
        player_y = position.get('y', '?')
        player_map = player_data.get('location', 'unknown')
        
        # Build objectives context (try to get from overall agent if available)
        objectives_context = "CURRENT GOAL:\nNavigate and interact to progress in the game."
        if hasattr(self, '_current_objective_context') and self._current_objective_context:
            objectives_context = f"CURRENT GOAL:\n{self._current_objective_context}"
        
        # Check for active dialogue
        dialogue_active = detect_dialogue(frame, threshold=0.45)
        
        # Build dialogue context
        if dialogue_active:
            # Check dialogues.json for recent dialogue info
            dialogue_info = self._get_recent_dialogue_info()
            if dialogue_info:
                dialogue_context = f"""DIALOGUE STATUS:
Dialogue is active.
Recent dialogue text: "{dialogue_info.get('dialogue_text', 'Unknown')}"
Dialogue started at: Map '{dialogue_info.get('map', 'unknown')}', NPC position ({dialogue_info.get('npc_x', '?')}, {dialogue_info.get('npc_y', '?')})

You are currently in a dialogue. Press 'A' to advance through the conversation."""
            else:
                dialogue_context = "DIALOGUE STATUS:\nDialogue is active. Press 'A' to advance through the conversation."
        else:
            dialogue_context = "DIALOGUE STATUS:\nNo active dialogue detected."
        
        # Build prompt
        prompt = INTERACTION_PROMPT.format(
            target_description=target_description,
            objectives_context=objectives_context,
            dialogue_context=dialogue_context,
            player_x=player_x,
            player_y=player_y,
            player_map=player_map
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
    
    def _get_recent_dialogue_info(self) -> Optional[Dict]:
        """
        Get information about the most recent dialogue from dialogues.json.
        
        Returns:
            Dict with dialogue info (dialogue_text, map, npc position, event) or None
        """
        try:
            dialogue_path = Path(DIALOGUE_LOG_FILE)
            if not dialogue_path.exists():
                return None
            
            with open(dialogue_path, 'r') as f:
                dialogues = json.load(f)
            
            if not dialogues:
                return None
            
            # Get most recent dialogue_start event
            recent = None
            for dialogue in reversed(dialogues):
                if dialogue.get('event') == 'dialogue_start':
                    recent = dialogue
                    break
            
            if not recent:
                return None
            
            # Extract relevant info using actual JSON fields
            npc_pos = recent.get('npc_tile_position', {})
            return {
                'dialogue_text': recent.get('dialogue_text', 'No text available'),
                'map': recent.get('player_map', 'unknown'),
                'npc_x': npc_pos.get('x', '?'),
                'npc_y': npc_pos.get('y', '?'),
                'event': recent.get('event', 'unknown')
            }
        except Exception as e:
            logger.debug(f"Could not load recent dialogue info: {e}")
            return None
    
    def _check_recent_dialogue_completion(self, player_x: int, player_y: int, player_map: str) -> Optional[Dict]:
        """
        Check if a dialogue just completed at the current player position within the last 5 steps.
        
        Args:
            player_x: Current player X position
            player_y: Current player Y position
            player_map: Current player map
            
        Returns:
            Dict with dialogue info if recent completion found, None otherwise
        """
        try:
            dialogue_path = Path(DIALOGUE_LOG_FILE)
            if not dialogue_path.exists():
                return None
            
            with open(dialogue_path, 'r') as f:
                dialogues = json.load(f)
            
            if not dialogues:
                return None
            
            # Check last 5 dialogue entries for dialogue_end at current position
            for dialogue in reversed(dialogues[-5:]):
                if dialogue.get('event') != 'dialogue_end':
                    continue
                
                # Check if dialogue_end is at current player position and map
                end_pos = dialogue.get('player_position', {})
                end_map = dialogue.get('player_map')
                
                if (end_pos.get('x') == player_x and 
                    end_pos.get('y') == player_y and 
                    end_map == player_map):
                    
                    # Find the corresponding dialogue_start
                    dialogue_count = dialogue.get('dialogue_count', 0)
                    for start_dialogue in reversed(dialogues):
                        if (start_dialogue.get('event') == 'dialogue_start' and
                            start_dialogue.get('dialogue_count', -1) == dialogue_count - 1):
                            return start_dialogue
                    
                    # If no matching start found, return basic info
                    return {'player_map': end_map, 'player_position': end_pos}
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not check recent dialogue completion: {e}")
            return None
    
    def _log_completed_dialogue_interaction(self, npc_pos: tuple, map_location: str, dialogue_info: Dict) -> None:
        """
        Log completed dialogue to interactions.json.
        
        Args:
            npc_pos: NPC position (x, y)
            map_location: Map location
            dialogue_info: Dialogue info dict from dialogues.json
        """
        try:
            log_path = Path(INTERACTION_LOG_FILE)
            
            # Load existing log
            if log_path.exists():
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Get navigation target info if available
            navigation_target = None
            if self.nav_state.current_target:
                navigation_target = {
                    'description': self.nav_state.current_target.description,
                    'target_pos': [self.nav_state.current_target.map_tile_position[0], self.nav_state.current_target.map_tile_position[1]],
                    'entity_type': self.nav_state.current_target.entity_type
                }
            
            # Add completed dialogue entry
            log_data.append({
                'action': 'DIALOGUE_COMPLETED',
                'tile_pos': list(npc_pos),
                'map_location': map_location,
                'result': 'dialogue_completed',
                'is_npc': True,
                'navigation_target': navigation_target,
                'dialogue_text': dialogue_info.get('dialogue_text', 'Unknown'),
                'timestamp': str(Path(INTERACTION_LOG_FILE).stat().st_mtime if log_path.exists() else 0)
            })
            
            # Save log
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Logged completed dialogue at {npc_pos} on {map_location}")
            
        except Exception as e:
            logger.error(f"Failed to log completed dialogue: {e}")
    
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
        
        if target_type in ["door", "stairs", "clock"]:
            # Warp points (doors, stairs, clocks): reached when map transitions
            # Reason: Stepping on warp causes immediate map transition
            if map_changed:
                logger.info(f"Warp target '{target_description}' reached - map transitioned from '{target_source_map}' to '{current_map}'")
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
            # Objects (NPCs, items): adjacent (distance == 1) for interaction
            # NPCs are non-traversable, so we need to be adjacent to interact
            reached = distance == 1
            if reached:
                logger.info(f"Object target '{target_description}' reached at adjacent distance {distance}")
            elif distance == 0:
                logger.warning(f"Object target at distance 0 - should be non-traversable, marking as reached anyway")
                reached = True
            return reached
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
    
    def _prepare_npc_updates_payload(self) -> Optional[Dict]:
        """
        Prepare the NPC updates payload for the client to send to server.
        
        Returns:
            Dict containing agent_grid, player_position, and player_map, or None if invalid
        """
        if not self.traversability_map:
            return None
        
        if self.player_map_tile_x is None or self.player_map_tile_y is None:
            logger.warning("Cannot prepare NPC updates - player position unknown")
            return None
        
        if self.player_map is None:
            logger.warning("Cannot prepare NPC updates - player map unknown")
            return None
        
        try:
            # Prepare payload with agent's 15x15 grid and player position
            payload = {
                "agent_grid": self.traversability_map,  # 15x15 grid with 'P' at (7,7) and NPCs as 'N'
                "player_position": {
                    "x": self.player_map_tile_x,
                    "y": self.player_map_tile_y
                },
                "player_map": self.player_map
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Error preparing NPC updates payload: {e}")
            return None
    
    def _create_debug_traversability_overlay(self, frame: np.ndarray, traversability_map: List[List[str]], navigation_targets: Optional[List] = None, state_data: Optional[dict] = None, filename_suffix: str = "_update", chosen_target: Optional[NavigationTarget] = None) -> Optional[List[List[str]]]:
        """
        Create debug overlay image showing side-by-side comparison: base traversability map (left) vs updated with NPCs (right).
        Creates a side-by-side image with navigation targets overlaid on the updated map.
        
        Args:
            frame: Game frame as numpy array
            traversability_map: 15x15 base traversability grid (without NPCs)
            navigation_targets: List of NavigationTarget objects to draw bounding boxes for
            state_data: Game state data (used to generate updated grid with NPCs)
            filename_suffix: Suffix for filename ("_update" or "_combined")
            chosen_target: The chosen NavigationTarget to highlight with blue box and cross
            
        Returns:
            Updated traversability grid with NPCs integrated, or None on error
        """
        try:
            # Find next incrementing index based on suffix
            debug_dir = Path("debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            existing_debugs = list(debug_dir.glob(f"debug_traversability_overlay_*{filename_suffix}.png"))
            prefix = "debug_traversability_overlay_"
            
            next_index = 0
            if existing_debugs:
                indices = []
                for path in existing_debugs:
                    try:
                        stem = path.stem  # e.g., "debug_traversability_overlay_003_update"
                        # Remove prefix and suffix to get the number
                        # e.g., "debug_traversability_overlay_003_update" -> "003"
                        if stem.startswith(prefix) and stem.endswith(filename_suffix):
                            # Remove prefix from start
                            after_prefix = stem[len(prefix):]  # "003_update"
                            # Remove suffix from end
                            num_str = after_prefix[:len(after_prefix) - len(filename_suffix)]  # "003"
                            indices.append(int(num_str))
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping file {path.name}: {e}")
                        continue
                if indices:
                    next_index = max(indices) + 1
            
            # Base grid is the input traversability_map (without NPCs)
            base_grid = traversability_map
            
            # Debug: Print traversability map before NPC integration
            logger.debug("=== TRAVERSABILITY MAP BEFORE NPC INTEGRATION ===")
            for i, row in enumerate(base_grid):
                logger.debug(f"Row {i:2d}: {''.join(row)}")
            
            # Create updated grid with NPC detections
            updated_grid = get_player_centered_grid(
                map_data=state_data.get('map', {}),
                fallback_grid=[['.' for _ in range(15)] for _ in range(15)],
                npc_detections=getattr(self, 'latest_npc_detections', None)
            )
            
            # Debug: Print traversability map after NPC integration
            logger.debug("=== TRAVERSABILITY MAP AFTER NPC INTEGRATION ===")
            for i, row in enumerate(updated_grid):
                logger.debug(f"Row {i:2d}: {''.join(row)}")
            
            # Create side-by-side image (15x15 tiles * 32px = 480x480 per side)
            grid_width, grid_height = 15 * 32, 15 * 32
            combined_image = Image.new('RGB', (grid_width * 2 + 4, grid_height), (0, 0, 0))
            
            # Preprocess frame: upscale x2 if frame is too small
            frame_image = Image.fromarray(frame.astype('uint8'), 'RGB')
            if frame_image.size[1] == 160:
                frame_image = frame_image.resize((480, 320), Image.NEAREST)
            if frame_image.size[1] == 320:
                # Pad the frame with 16px top and bottom only
                padded_height = frame_image.height + 32
                padded_frame = Image.new('RGB', (frame_image.width, padded_height), (0, 0, 0))
                padded_frame.paste(frame_image, (0, 16))
                frame_image = padded_frame
            
            # Now frame_image should be (480, 352)
            # Player is at (7, 5) in frame coordinates (32x32 tiles)
            # Player is at (7, 7) in grid coordinates (15x15 tiles)
            # Calculate offset to align player positions
            # Frame player pixel position: (7 * 32 + 16, 5 * 32 + 16) = (240, 176)
            # Grid player pixel position: (7 * 32 + 16, 7 * 32 + 16) = (240, 240)
            # Offset needed: grid_y - frame_y = 240 - 176 = 64 pixels (2 tiles down)
            player_offset_y = (7 - 5) * 32  # 64 pixels
            
            # Create background for side-by-side (with frame positioned to align player)
            # Left side: paste frame with offset
            left_bg = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
            left_bg.paste(frame_image, (0, player_offset_y))
            combined_image.paste(left_bg, (0, 0))
            
            # Right side: paste frame with same offset
            right_bg = Image.new('RGB', (grid_width, grid_height), (0, 0, 0))
            right_bg.paste(frame_image, (0, player_offset_y))
            combined_image.paste(right_bg, (grid_width + 4, 0))
            
            # Draw base grid on left side (original traversability map) over the frame
            left_overlay = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))
            self._draw_traversability_grid(left_overlay, base_grid, "BASE", None)
            combined_image.paste(left_overlay, (0, 0), left_overlay)
            
            # Draw vertical red separator line
            draw_separator = ImageDraw.Draw(combined_image, 'RGBA')
            draw_separator.line([(grid_width, 0), (grid_width, grid_height)], fill=(255, 0, 0, 255), width=4)
            
            # Draw updated grid on right side with navigation targets over the frame
            right_overlay = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))
            self._draw_traversability_grid(right_overlay, updated_grid, "AFTER", navigation_targets, chosen_target)
            combined_image.paste(right_overlay, (grid_width + 4, 0), right_overlay)
            
            # Add grid axes with map tile coordinates using PIL ImageDraw
            # Create a larger canvas with margins for axes
            margin_left = 40
            margin_top = 30
            margin_right = 40
            margin_bottom = 20
            
            canvas_width = combined_image.width + margin_left + margin_right
            canvas_height = combined_image.height + margin_top + margin_bottom
            canvas_image = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
            
            # Paste the combined image onto the canvas
            canvas_image.paste(combined_image, (margin_left, margin_top))
            
            # Draw grid lines and labels
            draw = ImageDraw.Draw(canvas_image)
            
            # Try to load a font for labels
            try:
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
            except:
                try:
                    label_font = ImageFont.load_default()
                except:
                    label_font = None
            
            # Add vertical grid lines and column numbers (map tile X coordinates)
            if self.player_map_tile_x is not None:
                # Left side columns (BEFORE)
                for col in range(16):  # 0 to 15 inclusive
                    x = margin_left + col * 32
                    # Draw vertical grid line
                    draw.line([(x, margin_top), (x, margin_top + grid_height)], 
                             fill=(0, 255, 255, 35), width=1)
                    
                    if col < 15:
                        # Calculate map tile X coordinate (player is at local tile 7)
                        map_tile_x = self.player_map_tile_x + (col - 7)
                        label = str(map_tile_x)
                        # Draw label at top
                        if label_font:
                            bbox = draw.textbbox((0, 0), label, font=label_font)
                            text_width = bbox[2] - bbox[0]
                            draw.text((x + 16 - text_width//2, margin_top - 20), label, 
                                     fill=(25, 25, 25), font=label_font)
                
                # Right side columns (AFTER) - offset by grid_width + 4
                offset_x = margin_left + grid_width + 4
                for col in range(16):
                    x = offset_x + col * 32
                    # Draw vertical grid line
                    draw.line([(x, margin_top), (x, margin_top + grid_height)], 
                             fill=(0, 255, 255, 35), width=1)
                    
                    if col < 15:
                        map_tile_x = self.player_map_tile_x + (col - 7)
                        label = str(map_tile_x)
                        if label_font:
                            bbox = draw.textbbox((0, 0), label, font=label_font)
                            text_width = bbox[2] - bbox[0]
                            draw.text((x + 16 - text_width//2, margin_top - 20), label, 
                                     fill=(25, 25, 25), font=label_font)
            
            # Add horizontal grid lines and row numbers (map tile Y coordinates)
            if self.player_map_tile_y is not None:
                for row in range(16):  # 0 to 15 inclusive
                    y = margin_top + row * 32
                    # Draw horizontal grid line
                    draw.line([(margin_left, y), (margin_left + grid_width * 2 + 4, y)], 
                             fill=(0, 255, 255, 35), width=1)
                    
                    if row < 15:
                        # Calculate map tile Y coordinate (player is at local tile 7)
                        map_tile_y = self.player_map_tile_y + (row - 7)
                        label = str(map_tile_y)
                        
                        # Left side row numbers
                        if label_font:
                            bbox = draw.textbbox((0, 0), label, font=label_font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                            draw.text((margin_left - text_width - 5, y + 16 - text_height//2), label, 
                                     fill=(25, 25, 25), font=label_font)
                            
                            # Right side row numbers
                            draw.text((margin_left + grid_width * 2 + 10, y + 16 - text_height//2), label, 
                                     fill=(25, 25, 25), font=label_font)
            
            # Save the final canvas image
            filename = f"{prefix}{next_index:03d}{filename_suffix}.png"
            canvas_image.save(debug_dir / filename)
            logger.debug(f"Saved traversability overlay with axes: {filename}")
            
            # Return updated grid for caller to save
            return updated_grid
            
        except Exception as e:
            logger.error(f"Failed to create debug traversability overlay: {e}")
            return None
    
    def _export_targets_json(self, targets: List[NavigationTarget], chosen_target: Optional[NavigationTarget] = None, filename_suffix: str = "_chosen") -> None:
        """
        Export navigation targets to JSON file with color mappings and bounding box info.
        
        Args:
            targets: List of NavigationTarget objects
            chosen_target: The chosen target (will be marked in JSON)
            filename_suffix: Suffix for filename (default "_chosen")
        """
        import json
        
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            
            # Get next index from existing JSON files
            prefix = "debug_traversability_overlay_"
            existing = sorted([f for f in debug_dir.glob(f"{prefix}*{filename_suffix}.json")])
            if existing:
                # Extract numeric part by removing prefix and suffix
                last_file = existing[-1].stem
                # Remove prefix
                after_prefix = last_file[len(prefix):]
                # Find where suffix starts (if any)
                for suffix in ["_update", "_combined", "_chosen"]:
                    if after_prefix.endswith(suffix):
                        after_prefix = after_prefix[:-len(suffix)]
                        break
                try:
                    next_index = int(after_prefix) + 1
                except ValueError:
                    next_index = 0
            else:
                next_index = 0
            
            # Type to color mapping (matching the visualization)
            type_color_map = {
                'npc': 'cyan',
                'door': 'yellow',
                'stairs': 'yellow',
                'clock': 'yellow',
                'object': 'orange',
                'boundary': 'magenta'
            }
            
            # Build JSON structure grouped by type
            output = {
                "metadata": {
                    "total_targets": len(targets),
                    "chosen_target_id": chosen_target.id if chosen_target else None,
                    "chosen_target_index": next(
                        (i for i, t in enumerate(targets) if t.id == chosen_target.id),
                        None
                    ) if chosen_target else None
                },
                "targets_by_type": {}
            }
            
            # Group targets by type and color
            for i, target in enumerate(targets):
                target_type = target.type
                color = type_color_map.get(target_type, 'magenta')
                type_color_key = f"{target_type}_{color}"
                
                if type_color_key not in output["targets_by_type"]:
                    output["targets_by_type"][type_color_key] = []
                
                # Build target info
                target_info = {
                    "index": i,
                    "id": target.id,
                    "description": target.description,
                    "map_tile_position": target.map_tile_position,
                    "local_tile_position": target.local_tile_position,
                    "priority": target.priority,
                    "entity_type": target.entity_type,
                    "source_map_location": target.source_map_location,
                    "tile_size": target.tile_size,
                    "is_chosen": (chosen_target and target.id == chosen_target.id)
                }
                
                # Add detected object bbox info if available
                if target.detected_object:
                    detected_obj = target.detected_object
                    target_info["detected_object"] = {
                        "name": detected_obj.name,
                        "confidence": detected_obj.confidence,
                        "bbox": detected_obj.bbox,  # {x, y, w, h}
                        "center_pixel": detected_obj.center_pixel,
                        "entity_type": detected_obj.entity_type,
                        "source": detected_obj.source
                    }
                
                output["targets_by_type"][type_color_key].append(target_info)
            
            # Save JSON file
            filename = f"{prefix}{next_index:03d}{filename_suffix}.json"
            json_path = debug_dir / filename
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.debug(f"Exported targets JSON: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export targets JSON: {e}")
    
    def _draw_traversability_grid(self, image: Image.Image, grid: List[List[str]], label: str, navigation_targets: Optional[List] = None, chosen_target: Optional[NavigationTarget] = None) -> None:
        """
        Draw traversability grid symbols on the image overlay.
        
        Args:
            image: PIL Image to draw on (480x480 for 15x15 grid)
            grid: 15x15 traversability grid
            label: Label for the overlay (BEFORE/AFTER)
            navigation_targets: List of NavigationTarget objects to draw bounding boxes for
            chosen_target: The chosen NavigationTarget to highlight with blue box and cross
        """
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # Try to load a small font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Grid dimensions: 15x15 tiles * 32px = 480x480 pixels
        tile_width, tile_height = 32, 32
        
        # Draw grid
        for y in range(15):
            for x in range(15):
                tile_symbol = grid[y][x]
                
                # Calculate pixel position (center of tile)
                pixel_x = x * tile_width + tile_width // 2
                pixel_y = y * tile_height + tile_height // 2
                
                # Choose color based on tile type
                if tile_symbol == 'P':
                    color = (255, 255, 0, 200)  # Yellow for player
                    outline_color = (255, 255, 0, 255)
                elif tile_symbol == 'N':
                    color = (255, 0, 0, 200)    # Red for NPCs
                    outline_color = (255, 0, 0, 255)
                    # Draw red box around NPC tiles (32x32 pixels)
                    box_x1 = x * tile_width
                    box_y1 = y * tile_height
                    box_x2 = box_x1 + 32
                    box_y2 = box_y1 + 32
                    draw.rectangle([box_x1, box_y1, box_x2, box_y2], 
                                 outline=(255, 0, 0, 255), width=2)
                elif tile_symbol in ['#', 'W', 'I']:
                    color = (128, 128, 128, 180)  # Gray for walls/obstacles
                    outline_color = (128, 128, 128, 255)
                elif tile_symbol == '~':
                    color = (0, 0, 255, 180)     # Blue for grass
                    outline_color = (0, 0, 255, 255)
                elif tile_symbol == '.':
                    color = (0, 255, 0, 150)     # Green for walkable
                    outline_color = (0, 255, 0, 255)
                else:
                    color = (255, 255, 255, 150)  # White for other
                    outline_color = (255, 255, 255, 255)
                
                # Draw background circle
                radius = 12  # Increased for 32x32 tiles
                draw.ellipse(
                    [(pixel_x - radius, pixel_y - radius), (pixel_x + radius, pixel_y + radius)],
                    fill=color,
                    outline=outline_color,
                    width=1
                )
                
                # Draw tile symbol
                bbox = draw.textbbox((0, 0), tile_symbol, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                draw.text(
                    (pixel_x - text_width//2, pixel_y - text_height//2),
                    tile_symbol,
                    fill=(0, 0, 0, 255),
                    font=font
                )
        
        # Draw navigation target bounding boxes if provided
        if navigation_targets:
            for i, target in enumerate(navigation_targets):
                try:
                    # Get target position in grid coordinates
                    grid_x, grid_y = target.local_tile_position
                    
                    # Use different colors for different target types
                    if target.type == 'npc':
                        box_color = (0, 255, 255, 255)  # Cyan for NPCs
                    elif target.type in ['door', 'stairs', 'clock']:
                        box_color = (255, 255, 0, 255)  # Yellow for warps
                    elif target.type == 'object':
                        box_color = (255, 165, 0, 255)  # Orange for objects
                    else:
                        box_color = (255, 0, 255, 255)  # Magenta for others
                    
                    # Check if we have a detected object with bbox info
                    if target.detected_object and target.detected_object.bbox:
                        # Use actual detected object bounding box (already in frame pixel coordinates)
                        bbox = target.detected_object.bbox
                        box_x1 = bbox['x']
                        box_y1 = bbox['y']
                        box_x2 = box_x1 + bbox['w']
                        box_y2 = box_y1 + bbox['h']
                        
                        # Adjust for frame offset (player alignment: 64px down)
                        player_offset_y = (7 - 5) * 32  # 64 pixels
                        
                        # # For non-uniquetiles_match sources, adjust bounding box and add extra offset
                        # if target.detected_object.source != "uniquetiles_match":
                        #     # Calculate bounding box with x2 adjustment (extend width and height)
                        #     box_x2 = box_x1 + (bbox['w'] * 2)
                        #     box_y2 = box_y1 + (bbox['h'] * 2)
                        #     # Move down by additional 16px
                        #     player_offset_y += 16
                        
                        box_y1 += player_offset_y
                        box_y2 += player_offset_y
                    else:
                        # Fallback: use tile-based sizing
                        # Use tile_size if available, otherwise default to 1x1
                        if target.tile_size:
                            tile_w, tile_h = target.tile_size
                        else:
                            tile_w, tile_h = 1, 1
                        
                        # Calculate pixel bounding box from tile position and size
                        box_x1 = grid_x * tile_width
                        box_y1 = grid_y * tile_height
                        box_x2 = box_x1 + (tile_w * tile_width)
                        box_y2 = box_y1 + (tile_h * tile_height)
                    
                    draw.rectangle([box_x1, box_y1, box_x2, box_y2], 
                                 outline=box_color, width=3)
                    
                    # Draw target index at center of bounding box
                    if font:
                        center_x = (box_x1 + box_x2) // 2
                        center_y = (box_y1 + box_y2) // 2
                        draw.text(
                            (center_x - 3, center_y - 3),
                            str(i),
                            fill=box_color,
                            font=font
                        )
                        
                except Exception as e:
                    logger.debug(f"Failed to draw bounding box for target {i}: {e}")
        
        # Draw chosen target with blue box and cross if provided
        if chosen_target:
            try:
                # Get chosen target position in grid coordinates
                grid_x, grid_y = chosen_target.local_tile_position
                
                # Direct mapping for 15x15 grid
                pixel_x = grid_x * tile_width + tile_width // 2
                pixel_y = grid_y * tile_height + tile_height // 2
                box_size = min(tile_width, tile_height) // 2  # Half tile size
                
                # Draw blue bounding box around chosen target
                box_x1 = pixel_x - box_size
                box_y1 = pixel_y - box_size
                box_x2 = pixel_x + box_size
                box_y2 = pixel_y + box_size
                
                blue_color = (0, 100, 255, 255)  # Blue for chosen target
                
                # Draw thicker box for chosen target
                draw.rectangle([box_x1, box_y1, box_x2, box_y2], 
                             outline=blue_color, width=4)
                
                # Draw cross through the box (X mark)
                draw.line([(box_x1, box_y1), (box_x2, box_y2)], fill=blue_color, width=3)
                draw.line([(box_x2, box_y1), (box_x1, box_y2)], fill=blue_color, width=3)
                
                logger.debug(f"Drew blue box and cross for chosen target at ({grid_x}, {grid_y})")
                
            except Exception as e:
                logger.debug(f"Failed to draw chosen target highlight: {e}")
        
        # Add label in top-left corner
        draw.rectangle([(0, 0), (80, 20)], fill=(0, 0, 0, 180))
        draw.text((5, 2), label, fill=(255, 255, 255, 255), font=font)
    
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
    
    def _resolve_target_overlaps(self, targets: List[NavigationTarget]) -> List[NavigationTarget]:
        """
        Resolve overlapping targets by keeping only one target per unique position.
        
        Filters out:
        - Objects larger than 4 tiles (2x2)
        - Targets overlapping with '#' (untraversable) in traversability map
          (Note: traversability map is already patched with untraversable active tiles)
        
        When conflicts occur, chooses the smaller object.
        
        Args:
            targets: List of navigation targets
            
        Returns:
            Filtered list with no position overlaps and no oversized objects
        """
        if not targets:
            return targets
        
        # Filter out oversized objects (larger than 2x2 tiles)
        filtered_targets = []
        for target in targets:
            if target.tile_size:
                tile_w, tile_h = target.tile_size
                tile_area = tile_w * tile_h
                if tile_area > 4:
                    logger.debug(f"Discarding oversized target ({tile_w}x{tile_h}): {target.description}")
                    continue
            filtered_targets.append(target)
        
        logger.info(f"After size filtering: {len(targets)} -> {len(filtered_targets)} targets")
        
        # Filter out targets that overlap with untraversable tiles in traversability map
        # (traversability map is already patched with untraversable active tiles via _patch_traversability_with_active_tiles)
        if self.traversability_map and self.player_map_tile_x is not None and self.player_map_tile_y is not None:
            untraversable_filtered = []
            for target in filtered_targets:
                tile_pos = target.map_tile_position
                # Convert map tile to local grid position (player at 7,7)
                local_x = 7 + (tile_pos[0] - self.player_map_tile_x)
                local_y = 7 + (tile_pos[1] - self.player_map_tile_y)
                
                # Check if position is within traversability map bounds
                if 0 <= local_y < 15 and 0 <= local_x < 15:
                    tile_value = self.traversability_map[local_y][local_x]
                    if tile_value == '#':
                        logger.debug(f"Discarding target overlapping with untraversable tile '#' at {tile_pos} (local: {local_x},{local_y}): {target.description}")
                        continue
                
                untraversable_filtered.append(target)
            
            logger.info(f"After traversability filtering (includes active tiles): {len(filtered_targets)} -> {len(untraversable_filtered)} targets")
            filtered_targets = untraversable_filtered
        
        # Group targets by position
        position_groups = {}
        for target in filtered_targets:
            pos_key = (target.map_tile_position, target.source_map_location)
            if pos_key not in position_groups:
                position_groups[pos_key] = []
            position_groups[pos_key].append(target)
        
        resolved_targets = []
        
        for pos_key, group_targets in position_groups.items():
            if len(group_targets) == 1:
                # No overlap, keep the single target
                resolved_targets.append(group_targets[0])
            else:
                # Multiple targets at same position, choose smallest
                logger.debug(f"Resolving overlap at {pos_key}: {len(group_targets)} targets")
                
                # Sort by size (smallest first), using tile area
                # Targets without tile_size get area of 1 (highest priority)
                sorted_targets = sorted(
                    group_targets,
                    key=lambda t: (t.tile_size[0] * t.tile_size[1]) if t.tile_size else 1
                )
                
                # Keep the smallest target
                chosen = sorted_targets[0]
                resolved_targets.append(chosen)
                
                tile_size_str = f"{chosen.tile_size[0]}x{chosen.tile_size[1]}" if chosen.tile_size else "unknown"
                logger.debug(f"Kept smallest target ({tile_size_str}): {chosen.description}")
        
        logger.info(f"Final resolved targets: {len(filtered_targets)} -> {len(resolved_targets)}")
        return resolved_targets
    
    # ==================== Step 6 & 7: Stuck Detection Methods ====================
    
    def _get_completed_npc_dialogues(self) -> List[Dict]:
        """
        Get list of completed NPC dialogues from dialogue log.
        
        Returns:
            List of dicts with 'npc_tile_position', 'player_map', 'active_tile_filename'
        """
        dialogue_log_path = Path(DIALOGUE_LOG_FILE)
        
        if not dialogue_log_path.exists():
            return []
        
        try:
            with open(dialogue_log_path, 'r') as f:
                log_data = json.load(f)
            
            # Find all completed dialogues (has both dialogue_start and dialogue_end events)
            dialogue_counts = {}
            for entry in log_data:
                dialogue_count = entry.get('dialogue_count')
                if dialogue_count is None:
                    continue
                
                if dialogue_count not in dialogue_counts:
                    dialogue_counts[dialogue_count] = {'start': None, 'end': None}
                
                event_type = entry.get('event')
                if event_type == 'dialogue_start':
                    dialogue_counts[dialogue_count]['start'] = entry
                elif event_type == 'dialogue_end':
                    dialogue_counts[dialogue_count]['end'] = entry
            
            # Extract completed NPC dialogues
            completed_npcs = []
            for dialogue_count, entries in dialogue_counts.items():
                if entries['start'] and entries['end']:
                    start_entry = entries['start']
                    # Only include if it's an NPC dialogue (has active_tile_class='npc')
                    if start_entry.get('active_tile_class') == 'npc':
                        completed_npcs.append({
                            'npc_tile_position': start_entry.get('npc_tile_position', {}),
                            'player_map': start_entry.get('player_map', 'unknown'),
                            'active_tile_filename': start_entry.get('active_tile_filename', ''),
                            'dialogue_count': dialogue_count
                        })
            
            logger.info(f"Found {len(completed_npcs)} completed NPC dialogues")
            return completed_npcs
            
        except Exception as e:
            logger.error(f"Failed to load completed NPC dialogues: {e}")
            return []
    
    def _get_completed_npc_interactions(self) -> List[Dict]:
        """
        Get list of completed NPC interactions from interactions.json.
        
        Returns:
            List of dicts with 'tile_pos' and 'map_location' for NPCs that have been interacted with
        """
        interactions_path = Path(INTERACTION_LOG_FILE)
        
        if not interactions_path.exists():
            return []
        
        try:
            with open(interactions_path, 'r') as f:
                interactions = json.load(f)
            
            # Filter for NPC interactions only
            npc_interactions = [
                {
                    'tile_pos': tuple(interaction.get('tile_pos', [])),
                    'map_location': interaction.get('map_location', '')
                }
                for interaction in interactions
                if interaction.get('is_npc', False) and interaction.get('result') == 'dialogue_started'
            ]
            
            logger.debug(f"Loaded {len(npc_interactions)} completed NPC interactions")
            return npc_interactions
            
        except Exception as e:
            logger.error(f"Failed to load completed NPC interactions: {e}")
            return []
    
    def _ensure_cache_directories(self) -> None:
        """Ensure navigation cache directories exist."""
        cache_dir = Path(NAVIGATION_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create interaction logs directory
        interaction_dir = Path(INTERACTION_LOG_FILE).parent
        interaction_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache directories initialized: {NAVIGATION_CACHE_DIR}")
    
    # NOTE: seen_tiles_cache.json is currently not being used
    # def _load_seen_tiles_cache(self) -> None:
    #     """Load seen tiles cache from disk."""
    #     cache_path = Path(NAVIGATION_CACHE_DIR) / "seen_tiles_cache.json"
    #     if cache_path.exists():
    #         try:
    #             with open(cache_path, 'r') as f:
    #                 data = json.load(f)
    #                 # Convert list of [x, y] to tuples and keep sorted
    #                 self.seen_tiles_cache = {
    #                     map_name: sorted([tuple(pos) for pos in positions])
    #                     for map_name, positions in data.items()
    #                 }
    #             total_tiles = sum(len(positions) for positions in self.seen_tiles_cache.values())
    #             logger.info(f"Loaded seen tiles cache: {len(self.seen_tiles_cache)} maps, {total_tiles} tiles")
    #         except Exception as e:
    #             logger.error(f"Failed to load seen tiles cache: {e}")
    #             self.seen_tiles_cache = {}
    #     else:
    #         self.seen_tiles_cache = {}
    #         logger.info("No existing seen tiles cache found")
    
    # NOTE: seen_tiles_cache.json is currently not being used
    # def _save_seen_tiles_cache(self) -> None:
    #     """Save seen tiles cache to disk."""
    #     cache_path = Path(NAVIGATION_CACHE_DIR) / "seen_tiles_cache.json"
    #     try:
    #         # Convert tuples to lists for JSON serialization
    #         data = {
    #             map_name: [list(pos) for pos in positions]
    #             for map_name, positions in self.seen_tiles_cache.items()
    #         }
    #         with open(cache_path, 'w') as f:
    #             json.dump(data, f, indent=2)
    #         logger.debug(f"Saved seen tiles cache: {len(self.seen_tiles_cache)} maps")
    #     except Exception as e:
    #         logger.error(f"Failed to save seen tiles cache: {e}")
    
    # NOTE: seen_tiles_cache.json is currently not being used
    # def _check_unseen_grass(self, state_data: dict, frame: np.ndarray) -> None:
    #     """
    #     Efficiently check for unseen grass tiles and update caches on the fly.
    #     
    #     Uses player displacement tracking to determine how many new tiles are visible.
    #     Only runs object detector if >80% of visible area is unexplored.
    #     
    #     Args:
    #         state_data: Game state data
    #         frame: Current game frame (480x352 padded)
    #     """
    #     current_pos = (self.player_map_tile_x, self.player_map_tile_y)
    #     current_map = self.player_map
    #     
    #     # Skip if position or map is unknown
    #     if None in current_pos or current_map is None:
    #         return
    #     
    #     # Initialize last check position if needed
    #     if self.last_grass_check_pos is None:
    #         self.last_grass_check_pos = current_pos
    #         self._detect_and_cache_grass(frame, current_pos, current_map, state_data)
    #         self._mark_position_as_seen(current_map, current_pos)
    #         return
    #     
    #     # Check if we moved to a new map
    #     map_changed = (self.nav_state.last_player_map and 
    #                   current_map != self.nav_state.last_player_map)
    #     
    #     # Calculate displacement since last check
    #     last_check_pos = self.last_grass_check_pos
    #     dx = current_pos[0] - last_check_pos[0]
    #     dy = current_pos[1] - last_check_pos[1]
    #     
    #     # Calculate number of new tiles visible (based on 15x9 visible area)
    #     # Player is at (7, 5) in frame (0-indexed), so visible tiles are:
    #     # X: player_x - 7 to player_x + 7 (15 tiles wide)
    #     # Y: player_y - 4 to player_y + 4 (9 tiles tall, excluding top/bottom rows)
    #     
    #     if map_changed:
    #         # Entire 15x9 area is potentially new
    #         new_tiles_count = 15 * 9
    #         logger.info(f"Map changed to {current_map}, checking full 15x9 area ({new_tiles_count} tiles)")
    #     else:
    #         # Calculate based on displacement
    #         # Horizontal movement: new_tiles = abs(dx) * 9 (9 tiles tall)
    #         # Vertical movement: new_tiles = abs(dy) * 15 (15 tiles wide)
    #         if dx != 0:
    #             new_tiles_count = abs(dx) * 9
    #         elif dy != 0:
    #             new_tiles_count = abs(dy) * 15
    #         else:
    #             # No movement, no new tiles
    #             return
    #         
    #         logger.debug(f"Player moved ({dx}, {dy}), estimated {new_tiles_count} new tiles")
    #     
    #     # Check what percentage of the visible area is unexplored
    #     visible_tiles = self._get_visible_tiles(current_pos)
    #     unexplored_count = self._count_unexplored_tiles(current_map, visible_tiles)
    #     unexplored_ratio = unexplored_count / len(visible_tiles) if visible_tiles else 0
    #     
    #     logger.debug(f"Unexplored ratio: {unexplored_ratio:.2%} ({unexplored_count}/{len(visible_tiles)} tiles)")
    #     
    #     # Run object detector if >80% unexplored
    #     if unexplored_ratio > 0.8:
    #         logger.info(f"Running grass detection ({unexplored_ratio:.2%} unexplored)")
    #         self._detect_and_cache_grass(frame, current_pos, current_map, state_data)
    #     
    #     # Mark current position as seen and update last check position
    #     self._mark_position_as_seen(current_map, current_pos)
    #     self.last_grass_check_pos = current_pos
    
    def _get_visible_tiles(self, player_pos: tuple) -> List[tuple]:
        """
        Get list of visible tile positions in 15x9 area around player.
        
        Player is at (7, 5) in frame coordinates, which translates to a 15x9 visible area
        (excluding top and bottom rows which are cropped).
        
        Args:
            player_pos: (x, y) player map tile position
            
        Returns:
            List of (x, y) map tile positions that are visible
        """
        visible = []
        player_x, player_y = player_pos
        
        # Visible range: X from -7 to +7 (15 tiles), Y from -4 to +4 (9 tiles)
        for dy in range(-4, 5):  # -4 to 4 inclusive (9 tiles)
            for dx in range(-7, 8):  # -7 to 7 inclusive (15 tiles)
                tile_x = player_x + dx
                tile_y = player_y + dy
                visible.append((tile_x, tile_y))
        
        return visible
    
    # NOTE: seen_tiles_cache.json is currently not being used
    # def _count_unexplored_tiles(self, map_name: str, tiles: List[tuple]) -> int:
    #     """
    #     Count how many tiles in the list have not been seen before on this map.
    #     
    #     Args:
    #         map_name: Map location name
    #         tiles: List of (x, y) tile positions to check
    #         
    #     Returns:
    #         Number of unseen tiles
    #     """
    #     if map_name not in self.seen_tiles_cache:
    #         return len(tiles)
    #     
    #     seen_set = set(self.seen_tiles_cache[map_name])
    #     unexplored = sum(1 for tile in tiles if tile not in seen_set)
    #     return unexplored
    # 
    # def _mark_position_as_seen(self, map_name: str, position: tuple) -> None:
    #     """
    #     Mark all tiles visible from this position as seen.
    #     
    #     Args:
    #         map_name: Map location name
    #         position: (x, y) player position
    #     """
    #     visible_tiles = self._get_visible_tiles(position)
    #     
    #     if map_name not in self.seen_tiles_cache:
    #         self.seen_tiles_cache[map_name] = []
    #     
    #     # Add new tiles and keep sorted
    #     existing_set = set(self.seen_tiles_cache[map_name])
    #     new_tiles = [tile for tile in visible_tiles if tile not in existing_set]
    #     
    #     if new_tiles:
    #         self.seen_tiles_cache[map_name].extend(new_tiles)
    #         self.seen_tiles_cache[map_name].sort()
    #         self._save_seen_tiles_cache()
    #         logger.debug(f"Marked {len(new_tiles)} new tiles as seen on {map_name}")
    
    def _replan_path_to_current_target(self, state_data: dict) -> None:
        """
        Recalculate A* path to current target with updated traversability map.
        
        Args:
            state_data: Game state data
        """
        if not self.nav_state.current_target:
            return
        
        logger.info(f"Replanning path to {self.nav_state.current_target.description}")
        
        # Plan new path with updated traversability
        new_path = self._plan_path_to_target(self.nav_state.current_target, state_data)
        
        if new_path:
            # Update navigation state with new path
            self.nav_state.current_path = new_path
            self.nav_state.path_index = 0
            self.nav_state.steps_since_progress = 0
            logger.info(f"Replanned path: {len(new_path)} actions")
        else:
            logger.warning("Failed to replan path - switching to interaction mode")
            self.nav_state.phase = "interacting"
            self.nav_state.reached_target_for_interaction = False
    
    def _detect_dialogue_state(self, state_data: dict, frame: np.ndarray) -> bool:
        """
        Detect if dialogue is currently active.
        
        Args:
            state_data: Game state data
            frame: Current game frame
            
        Returns:
            True if dialogue detected, False otherwise
        """
        # Use imported detect_dialogue from custom_utils.detectors
        return detect_dialogue(frame, threshold=0.45)
    
    def _check_interaction_result(self, state_data: dict, frame: np.ndarray) -> None:
        """
        Check the result of pressing 'A' button in previous step.
        
        Handles different interaction outcomes:
        - If map changed: tile was door/stairs, erase from cache
        - If dialogue opened: mark as NPC
        - If no dialogue and no map change: try walking to tile, if unreachable mark as untraversable
        
        This is called at the START of the next step after pressing 'A'.
        
        Args:
            state_data: Game state data
            frame: Current game frame
        """
        from custom_utils.detectors import detect_dialogue
        
        # Get player position and map
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        player_x = position.get('x')
        player_y = position.get('y')
        player_map = player_data.get('location')
        
        # Get the tile we tried to interact with
        interaction_tile = self.nav_state.expected_interaction_tile
        if not interaction_tile:
            logger.warning("No expected interaction tile set")
            self._reset_nav_state()
            self.nav_state.interaction_failed = True
            return
        
        # Check if map changed (indicates door/stairs)
        map_changed = (self.nav_state.last_player_map and 
                      player_map != self.nav_state.last_player_map)
        
        if map_changed:
            # Map changed - this was a door/stairs, erase from cache
            logger.info(f"🪜 Map changed after 'A' press - tile {interaction_tile} was door/stairs, erasing from cache")
            
            tile_filename = f"{self.nav_state.last_player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
            if tile_filename in self.active_tile_index:
                del self.active_tile_index[tile_filename]
                save_active_tile_index(self.active_tile_index)
                logger.debug(f"Erased {tile_filename} from active_tile_index")
            
            # Reset and continue (successful door/stairs interaction)
            self._reset_nav_state()
            self.nav_state.interaction_failed = False
            return
        
        # Check if dialogue is currently active
        dialogue_active = detect_dialogue(frame, threshold=0.45)
        
        if map_changed:
            # Map changed - this was a door/stairs, erase from cache
            logger.info(f"🪜 Map changed after 'A' press - tile {interaction_tile} was door/stairs, erasing from cache")
            
            tile_filename = f"{self.nav_state.last_player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
            if tile_filename in self.active_tile_index:
                del self.active_tile_index[tile_filename]
                save_active_tile_index(self.active_tile_index)
                logger.debug(f"Erased {tile_filename} from active_tile_index")
            
            # Reset and continue (successful door/stairs interaction)
            self._reset_nav_state()
            self.nav_state.interaction_failed = False
            return
        
        # Check if dialogue is currently active
        dialogue_active = detect_dialogue(frame, threshold=0.45)
        
        if dialogue_active:
            # Dialogue opened - update class to NPC
            logger.info(f"💬 Dialogue detected after 'A' press - updating tile {interaction_tile} to NPC")
            
            tile_filename = f"{player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
            if tile_filename in self.active_tile_index:
                self.active_tile_index[tile_filename]['class'] = 'npc'
                save_active_tile_index(self.active_tile_index)
                logger.debug(f"Updated {tile_filename} class to npc")
            else:
                # Fallback: mark as NPC if not in cache
                mark_and_save_tile(
                    interaction_tile, frame, 'npc', self.active_tile_index, player_map,
                    allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                    player_map_tile_y=self.player_map_tile_y
                )
            
            log_interaction(
                action='A',
                tile_pos=interaction_tile,
                result='npc_dialogue_triggered',
                is_npc=True,
                map_location=player_map,
                navigation_target=None
            )
            
            # Keep in interacting phase to let VLM handle dialogue
        else:
            # No dialogue and no map change - test reachability
            logger.info(f"🔍 No dialogue after 'A' press - testing if tile {interaction_tile} is reachable")
            
            # Try to pathfind to the tile
            # Calculate local tile position relative to player
            local_x = 7 + (interaction_tile[0] - player_x)
            local_y = 7 + (interaction_tile[1] - player_y)
            
            test_target = NavigationTarget(
                id="test_interaction_target",
                type="object",
                map_tile_position=interaction_tile,
                local_tile_position=(local_x, local_y),
                description="interaction_test",
                entity_type="test",
                source_map_location=player_map
            )
            
            try:
                test_path = self._plan_path_to_target(test_target, state_data)
                path_length = len(test_path) if test_path else 0
                
                if path_length > 0 and path_length <= 2:  # Adjacent tile should have path length 1-2
                    # Tile is reachable - might be traversable, update class to traversable or keep as is
                    logger.info(f"✅ Tile {interaction_tile} is reachable (path length {path_length}) - keeping as traversable")
                    # Update class to indicate it's traversable
                    tile_filename = f"{player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
                    if tile_filename in self.active_tile_index:
                        # If it was marked as interacting, we can leave it or mark as traversable
                        # For now, just log that it's reachable
                        pass
                    
                    log_interaction(
                        action='A',
                        tile_pos=interaction_tile,
                        result='reachable_no_dialogue',
                        is_npc=False,
                        map_location=player_map,
                        navigation_target=None
                    )
                else:
                    # Tile is not reachable or too far - update class to untraversable
                    logger.info(f"❌ Tile {interaction_tile} is unreachable (path length {path_length}) - updating to untraversable")
                    tile_filename = f"{player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
                    if tile_filename in self.active_tile_index:
                        self.active_tile_index[tile_filename]['class'] = 'untraversable'
                        save_active_tile_index(self.active_tile_index)
                        logger.debug(f"Updated {tile_filename} class to untraversable")
                    else:
                        # Fallback: mark as untraversable if not in cache
                        mark_and_save_tile(
                            interaction_tile, frame, 'untraversable', self.active_tile_index, player_map,
                            allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                            player_map_tile_y=self.player_map_tile_y
                        )
                    
                    log_interaction(
                        action='A',
                        tile_pos=interaction_tile,
                        result='unreachable_non_traversable',
                        is_npc=False
                    )
            except Exception as e:
                logger.warning(f"Failed to test reachability for tile {interaction_tile}: {e}")
                # If pathfinding fails, assume untraversable
                tile_filename = f"{player_map}_{interaction_tile[0]}_{interaction_tile[1]}.png"
                if tile_filename in self.active_tile_index:
                    self.active_tile_index[tile_filename]['class'] = 'untraversable'
                    save_active_tile_index(self.active_tile_index)
                    logger.debug(f"Updated {tile_filename} class to untraversable due to pathfind error")
                else:
                    mark_and_save_tile(
                        interaction_tile, frame, 'untraversable', self.active_tile_index, player_map,
                        allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                        player_map_tile_y=self.player_map_tile_y
                    )
                
                log_interaction(
                    action='A',
                    tile_pos=interaction_tile,
                    result='pathfind_error_non_traversable',
                    is_npc=False
                )
            
            # Reset and start new navigation
            self._reset_nav_state()
            self.nav_state.interaction_failed = True
    
    def _track_movement_queue(self, action: str, current_pos: tuple) -> None:
        """
        Track movement actions and positions for stuck detection.
        
        Args:
            action: Action taken
            current_pos: Current (x, y) position
        """
        # Only track directional movements
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            self.nav_state.movement_queue.append(action)
            self.nav_state.position_queue.append(current_pos)
            
            # Keep only last 3
            if len(self.nav_state.movement_queue) > 3:
                self.nav_state.movement_queue.pop(0)
                self.nav_state.position_queue.pop(0)
    
    def _check_failed_movement(self, frame: np.ndarray, state_data: dict) -> None:
        """
        Step 7: Check if movement has failed 3 consecutive times.
        
        If same action 3 times with no position change, mark the blocked tile.
        Uses the last fired action direction to determine which tile to mark,
        which is more reliable than detected player facing direction.
        
        Args:
            frame: Current game frame
            state_data: Game state data
        """
        if len(self.nav_state.movement_queue) < 3:
            return
        
        # Check if all 3 actions are the same
        if not all(action == self.nav_state.movement_queue[0] 
                   for action in self.nav_state.movement_queue):
            return
        
        # Check if all 3 positions are the same
        if not all(pos == self.nav_state.position_queue[0] 
                   for pos in self.nav_state.position_queue):
            return
        
        # Failed movement detected!
        action = self.nav_state.movement_queue[0]
        current_pos = self.nav_state.position_queue[0]
        
        logger.warning(f"Failed movement detected: {action} x3 at {current_pos}")
        
        # Calculate blocked tile position based on LAST FIRED ACTION direction
        # This is more reliable than detected player facing direction
        action_offset = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0)
        }
        
        offset = action_offset.get(action, (0, 0))
        blocked_tile_x = current_pos[0] + offset[0]
        blocked_tile_y = current_pos[1] + offset[1]
        blocked_tile = (blocked_tile_x, blocked_tile_y)
        
        logger.info(f"Marking tile {blocked_tile} as untraversable (based on action '{action}' from {current_pos})")
        
        # Mark tile as non-traversable
        mark_and_save_tile(
            blocked_tile, frame, 'untraversable', self.active_tile_index, self.player_map,
            allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
            player_map_tile_y=self.player_map_tile_y
        )
        
        # Log interaction
        log_interaction(
            action=action,
            tile_pos=blocked_tile,
            result='blocked_after_3_attempts',
            is_npc=False,
            map_location=self.player_map,
            navigation_target=None
        )
        
        # Clear queues
        self.nav_state.movement_queue.clear()
        self.nav_state.position_queue.clear()
        
        # Replan path with updated traversability
        self._replan_path_to_current_target(state_data)
        
        # Increment failed movement count
        self.nav_state.failed_movement_count += 1
        logger.debug(f"Failed movement count: {self.nav_state.failed_movement_count}")
        
        # If too many failed movements, switch to interaction
        if self.nav_state.failed_movement_count >= 3:
            logger.warning(f"Failed movement count {self.nav_state.failed_movement_count} >= 3 - switching to interaction mode")
            self.nav_state.phase = "interacting"
            self.nav_state.reached_target_for_interaction = False
        
        logger.info(f"Cleared movement queues after marking tile {blocked_tile} as untraversable")
    
    def _handle_npc_initiated_dialogue(self, state_data: dict, frame: np.ndarray) -> None:
        """
        Handle NPC-initiated dialogue interruption during navigation.
        
        Removes the current navigation target from consideration by excluding
        its position since we never reached it.
        
        Args:
            state_data: Game state data
            frame: Current game frame
        """
        if self.nav_state.current_target:
            target_pos = self.nav_state.current_target.map_tile_position
            logger.warning(f"Removing unreached target at {target_pos} from consideration due to NPC interruption")
            
            # Add to exclusion list
            if not hasattr(self, '_exclude_positions'):
                self._exclude_positions = []
            self._exclude_positions.append(target_pos)
            
            # Clear current target and path
            self.nav_state.current_target = None
            self.nav_state.current_path = []
            self.nav_state.path_index = 0
    
    def _detect_and_log_npc_after_dialogue(self, state_data: dict, frame: np.ndarray) -> None:
        """
        Detect and log NPC position after dialogue completes.
        
        Checks for NPCs adjacent to player using object detector.
        If exactly one NPC detected, logs that NPC.
        If multiple or zero NPCs, falls back to player facing direction (if score > 0.8).
        
        Args:
            state_data: Game state data
            frame: Current game frame
        """
        if self.player_map_tile_x is None or self.player_map_tile_y is None:
            logger.warning("Cannot detect NPC after dialogue - player position unknown")
            return
        
        player_pos = (self.player_map_tile_x, self.player_map_tile_y)
        
        # Detect NPCs using object detector
        try:
            npc_detections = self.object_detector.detect_exact_tile_matches(frame, match_class="npc")
            
            # Filter for NPCs adjacent to player (Manhattan distance == 1)
            adjacent_npcs = []
            for detection in npc_detections:
                npc_x = detection.get('tile_position', {}).get('map_x')
                npc_y = detection.get('tile_position', {}).get('map_y')
                
                if npc_x is not None and npc_y is not None:
                    manhattan_dist = abs(npc_x - player_pos[0]) + abs(npc_y - player_pos[1])
                    if manhattan_dist == 1:
                        adjacent_npcs.append((npc_x, npc_y))
            
            logger.info(f"Found {len(adjacent_npcs)} adjacent NPCs after dialogue: {adjacent_npcs}")
            
            # Case 1: Exactly one adjacent NPC - use it
            if len(adjacent_npcs) == 1:
                npc_pos = adjacent_npcs[0]
                logger.info(f"Marking single adjacent NPC at {npc_pos}")
                mark_and_save_tile(
                    npc_pos, frame, 'npc', self.active_tile_index, self.player_map,
                    allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                    player_map_tile_y=self.player_map_tile_y
                )
                log_interaction(
                    action='NPC_initiated',
                    tile_pos=npc_pos,
                    result='npc_interrupted_navigation',
                    is_npc=True,
                    map_location=self.player_map,
                    navigation_target=None
                )
                return
            
            # Case 2: Multiple or zero NPCs - fall back to player facing direction
            from custom_utils.detectors import detect_player_direction
            
            # Use cached player direction result if available (passed from overall agent)
            if hasattr(self, '_player_direction_result') and self._player_direction_result:
                direction_result = self._player_direction_result
                logger.debug("Using cached player direction result from overall agent")
            else:
                # Detect player direction with score (fallback if not provided)
                direction_result = detect_player_direction(frame, match_threshold=0.7)
                logger.debug("Detecting player direction (not provided by overall agent)")
            
            if direction_result:
                direction, score = direction_result if isinstance(direction_result, tuple) else (direction_result, 0.0)
                
                logger.info(f"Player facing direction: {direction} (score: {score:.2f})")
                
                if score > 0.8:
                    # Use facing direction to determine NPC position
                    facing_offset = {
                        'North': (0, -1),
                        'South': (0, 1),
                        'East': (1, 0),
                        'West': (-1, 0)
                    }
                    
                    offset = facing_offset.get(direction, (0, 0))
                    npc_pos = (player_pos[0] + offset[0], player_pos[1] + offset[1])
                    
                    logger.info(f"Using player facing direction ({direction}) to mark NPC at {npc_pos}")
                    mark_and_save_tile(
                        npc_pos, frame, 'npc', self.active_tile_index, self.player_map,
                        allow_upgrade=False, player_map_tile_x=self.player_map_tile_x,
                        player_map_tile_y=self.player_map_tile_y
                    )
                    log_interaction(
                        action='NPC_initiated',
                        tile_pos=npc_pos,
                        result='npc_interrupted_navigation_facing',
                        is_npc=True,
                        map_location=self.player_map,
                        navigation_target=None
                    )
                else:
                    logger.warning(f"Player direction score too low ({score:.2f}) - cannot reliably detect NPC position")
            else:
                logger.warning("Could not detect player facing direction - cannot determine NPC position")
                
        except Exception as e:
            logger.error(f"Error detecting NPC after dialogue: {e}")
    
    def _get_facing_action_for_target(self) -> Optional[str]:
        """
        Determine the action needed to face the current target.
        
        Returns the button press ('UP', 'DOWN', 'LEFT', 'RIGHT') to face the target,
        or None if already facing it or target is not adjacent.
        
        Returns:
            Action string or None
        """
        if not self.nav_state.current_target:
            return None
        
        target_pos = self.nav_state.current_target.map_tile_position
        player_x = self.player_map_tile_x
        player_y = self.player_map_tile_y
        
        if player_x is None or player_y is None:
            return None
        
        # Calculate delta
        dx = target_pos[0] - player_x
        dy = target_pos[1] - player_y
        
        # Must be adjacent (Manhattan distance = 1) or on the same tile (distance = 0)
        distance = abs(dx) + abs(dy)
        if distance == 0:
            # On the same tile (e.g., door/stairs) - no facing needed
            return None
        elif distance == 1:
            # Adjacent - determine facing direction
            if dx == 1 and dy == 0:
                return 'RIGHT'  # Face East
            elif dx == -1 and dy == 0:
                return 'LEFT'   # Face West
            elif dx == 0 and dy == -1:
                return 'UP'     # Face North
            elif dx == 0 and dy == 1:
                return 'DOWN'   # Face South
            else:
                # Diagonal - shouldn't happen for adjacent targets
                return None
        else:
            # Too far - can't face directly
            return None
    
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
            logger.error(f"Failed to patch traversability map with active tiles: {e}")

