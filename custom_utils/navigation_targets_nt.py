"""
Navigation target generation from detected objects and traversability boundaries (NT version).

This version uses object_detector_nt.DetectedObject which supports the 'uniquetiles_match' source.
"""
from typing import List, Optional, Tuple, Literal, Dict
from pydantic import BaseModel
import logging

from custom_utils.object_detector_nt import DetectedObject
from custom_utils.coordinate_transform import (
    pixel_to_local_tile, local_tile_to_map_tile,
    FRAME_WIDTH, FRAME_HEIGHT, LOCAL_TILE_WIDTH, LOCAL_TILE_HEIGHT_VISIBLE
)

logger = logging.getLogger(__name__)


class NavigationTarget(BaseModel):
    """Represents a navigation target for the agent."""
    id: str
    type: Literal["object", "boundary", "door", "stairs"]
    map_tile_position: Tuple[int, int]
    local_tile_position: Tuple[int, int]
    description: str
    priority: float = 0.0
    entity_type: Optional[str] = None
    detected_object: Optional[DetectedObject] = None
    source_map_location: Optional[str] = None
    tile_size: Optional[Tuple[int, int]] = None
    reachable: Optional[bool] = None
    exit_direction: Optional[Tuple[int, int]] = None  # For doors: (dx, dy) indicating exit direction


def pixel_to_tile_size(pixel_width: int, pixel_height: int) -> Tuple[int, int]:
    """Convert pixel width and height to units of tiles."""
    tile_width = max(int(pixel_width / FRAME_WIDTH * LOCAL_TILE_WIDTH), 1)
    tile_height = max(int(pixel_height / FRAME_HEIGHT * LOCAL_TILE_HEIGHT_VISIBLE), 1)
    return tile_width, tile_height


def generate_navigation_targets(
    detected_objects: List[DetectedObject],
    traversability_map: List[List[str]],
    player_map_tile_pos: Tuple[int, int],
    player_map_location: Optional[str] = None,
    include_grass_targets: bool = False,
    portal_connections: Optional[List[Dict]] = None
) -> List[NavigationTarget]:
    """
    Generate navigation targets from detected objects and traversable boundaries.
    
    Args:
        detected_objects: List of objects detected by ObjectDetector
        traversability_map: 15x15 grid of traversability symbols
        player_map_tile_pos: Player's current map tile position (x, y)
        player_map_location: Player's current map location string (e.g., "LITTLEROOT_TOWN")
        include_grass_targets: Whether to include grass targets (default False)
        portal_connections: List of portal connection dicts to add as targets
        
    Returns:
        Combined list of object targets and boundary targets
    """
    
    targets = []
    
    # Filter detected objects if include_grass_targets is True
    if include_grass_targets:
        detected_objects = [obj for obj in detected_objects if obj.entity_type == 'grass']
    
    # Convert detected objects to navigation targets
    for i, obj in enumerate(detected_objects):
        # Convert pixel center to local tile position
        pixel_x, pixel_y = obj.center_pixel
        local_tile_x, local_tile_y = pixel_to_local_tile(pixel_x, pixel_y)
        
        # Convert pixel size to tile size
        tile_width, tile_height = pixel_to_tile_size(obj.bbox['w'], obj.bbox['h'])
        
        # Convert local tile to map tile position
        map_tile_x, map_tile_y = local_tile_to_map_tile(local_tile_x, local_tile_y, player_map_tile_pos[0], player_map_tile_pos[1])
        
        # Create navigation target
        target = NavigationTarget(
            id=f"object_{i}",
            type="object",
            map_tile_position=(map_tile_x, map_tile_y),
            local_tile_position=(local_tile_x, local_tile_y),
            description=f"{obj.name} centered at ({map_tile_x}, {map_tile_y}) of size {tile_width}x{tile_height}",
            priority=0.0,
            entity_type=obj.entity_type,
            detected_object=obj,
            source_map_location=player_map_location,
            tile_size=(tile_width, tile_height)
        )
        targets.append(target)
    
    # Find and add boundary targets
    boundary_groups = find_traversable_boundaries(traversability_map)
    for j, boundary_group in enumerate(boundary_groups):
        # Use the center position of the contiguous group as the representative position
        center_idx = len(boundary_group) // 2
        local_tile_x, local_tile_y = boundary_group[center_idx]
        
        # Convert local tile to map tile position
        map_tile_x, map_tile_y = local_tile_to_map_tile(local_tile_x, local_tile_y, player_map_tile_pos[0], player_map_tile_pos[1])
        
        # Determine boundary direction
        if local_tile_y == 2:
            direction = "North"
        elif local_tile_y == 12:
            direction = "South"
        elif local_tile_x == 0:
            direction = "West"
        else:  # local_tile_x == 14
            direction = "East"
        
        # Create navigation target for boundary group
        target = NavigationTarget(
            id=f"boundary_{j}",
            type="boundary",
            map_tile_position=(map_tile_x, map_tile_y),
            local_tile_position=(local_tile_x, local_tile_y),
            description=f"{direction} boundary exit centered at ({map_tile_x}, {map_tile_y}) of length {len(boundary_group)}",
            priority=0.0,
            entity_type=None,
            detected_object=None,
            source_map_location=player_map_location
        )
        targets.append(target)
    
    # Find and add warp point targets (doors and stairs)
    warp_positions = find_warp_points(traversability_map)
    for k, (local_tile_x, local_tile_y, tile_char) in enumerate(warp_positions):
        # Convert local tile to map tile position
        map_tile_x, map_tile_y = local_tile_to_map_tile(local_tile_x, local_tile_y, player_map_tile_pos[0], player_map_tile_pos[1])

        # Distinguish doors from stairs
        if tile_char == 'D':
            target_type = "door"
            warp_type = "Door"
            # Detect exit direction for doors
            exit_dir = detect_door_exit_direction(traversability_map, local_tile_x, local_tile_y)
            # TODO: Consider creating single target for door pairs with perpendicular exit
        else:  # 'S'
            target_type = "stairs"
            warp_type = "Stairs"
            exit_dir = None

        # Create navigation target for warp point
        target = NavigationTarget(
            id=f"{target_type}_{k}",
            type=target_type,
            map_tile_position=(map_tile_x, map_tile_y),
            local_tile_position=(local_tile_x, local_tile_y),
            description=f"{warp_type} at ({map_tile_x}, {map_tile_y}) of size 1x1",
            priority=0.0,
            entity_type=None,
            detected_object=None,
            source_map_location=player_map_location,
            tile_size=(1, 1),  # TODO: it might be the case that doors are contiguous, fix someday
            exit_direction=exit_dir
        )
        targets.append(target)
    
    # Check reachability of all targets
    targets = check_targets_reachability(targets, traversability_map)

    # Apply prioritization (currently a pass-through)
    targets = prioritize_targets(targets)

    # Add portal targets if provided
    if portal_connections:
        for m, connection in enumerate(portal_connections):
            from_pos = connection['from_pos']
            to_map = connection['to_map']
            # Convert map tile to local tile
            local_tile_x = from_pos[0] - player_map_tile_pos[0] + 7
            local_tile_y = from_pos[1] - player_map_tile_pos[1] + 7
            # Only include if within the 15x15 grid
            if 0 <= local_tile_x < 15 and 0 <= local_tile_y < 15:
                portal_target = NavigationTarget(
                    id=f"portal_{m}",
                    type="door",
                    map_tile_position=from_pos,
                    local_tile_position=(local_tile_x, local_tile_y),
                    description=f"Door at ({from_pos[0]}, {from_pos[1]}) in {player_map_location} to go to {to_map}",
                    priority=0.0,
                    entity_type="portal",
                    detected_object=None,
                    source_map_location=player_map_location,
                    tile_size=(1, 1)
                )
                targets.append(portal_target)

    return targets


def is_traversable_boundary(tile: str) -> bool:
    """
    Check if a tile is traversable for boundary.
    So we avoid stairs etc which are traversable but not for boundaries.
    To help contiguity grouping.
    
    TODO: Improve this function to handle more tile types in the future.
    
    Args:
        tile: Traversability symbol from the map
        
    Returns:
        True if traversable, False otherwise
    """
    return tile in {'.', '~'}


def find_traversable_boundaries(
    traversability_map: List[List[str]]
) -> List[List[Tuple[int, int]]]:
    """
    Find walkable positions at grid edges (screen boundaries).
    
    Only considers the sub box spanning y rows 2-12 inclusive (zero indexed).
    Groups contiguous traversable boundary tiles by direction (north, south, east, west).
    Each contiguous line of traversable tiles in one direction forms one group.
    
    Args:
        traversability_map: 15x15 grid of traversability symbols
        
    Returns:
        List of boundary groups, where each group is a list of contiguous grid positions
    """
    boundary_groups = []
    
    # Only consider y rows 2-12 inclusive (zero indexed)
    min_y = 2
    max_y = 12
    
    # Helper function to group contiguous positions
    def group_contiguous(positions: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Group contiguous positions into separate lists."""
        if not positions:
            return []
        
        # Sort positions to ensure contiguous detection works
        positions = sorted(positions)
        groups = []
        current_group = [positions[0]]
        
        for i in range(1, len(positions)):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]
            
            # Check if contiguous (adjacent in the relevant direction)
            if prev_pos[0] == curr_pos[0]:  # Same x, check y adjacency
                if curr_pos[1] == prev_pos[1] + 1:
                    current_group.append(curr_pos)
                else:
                    groups.append(current_group)
                    current_group = [curr_pos]
            elif prev_pos[1] == curr_pos[1]:  # Same y, check x adjacency
                if curr_pos[0] == prev_pos[0] + 1:
                    current_group.append(curr_pos)
                else:
                    groups.append(current_group)
                    current_group = [curr_pos]
            else:
                groups.append(current_group)
                current_group = [curr_pos]
        
        groups.append(current_group)
        return groups
    
    # Check North boundary (y = min_y) - includes all x positions
    north_positions = []
    for x in range(15):
        if is_traversable_boundary(traversability_map[min_y][x]):
            north_positions.append((x, min_y))
    boundary_groups.extend(group_contiguous(north_positions))
    
    # Check South boundary (y = max_y) - includes all x positions
    south_positions = []
    for x in range(15):
        if is_traversable_boundary(traversability_map[max_y][x]):
            south_positions.append((x, max_y))
    boundary_groups.extend(group_contiguous(south_positions))
    
    # Check West boundary (x = 0) - exclude corners to avoid duplicates
    west_positions = []
    for y in range(min_y + 1, max_y):
        if is_traversable_boundary(traversability_map[y][0]):
            west_positions.append((0, y))
    boundary_groups.extend(group_contiguous(west_positions))
    
    # Check East boundary (x = 14) - exclude corners to avoid duplicates
    east_positions = []
    for y in range(min_y + 1, max_y):
        if is_traversable_boundary(traversability_map[y][14]):
            east_positions.append((14, y))
    boundary_groups.extend(group_contiguous(east_positions))
    
    return boundary_groups


def find_warp_points(
    traversability_map: List[List[str]]
) -> List[Tuple[int, int, str]]:
    """
    Find warp points (doors and stairs) in the traversability map.

    Searches for 'D' (door) and 'S' (stairs) tiles.
    Only considers y rows 2-12 inclusive (zero indexed).

    Args:
        traversability_map: 15x15 grid of traversability symbols

    Returns:
        List of tuples (x, y, tile_char) where tile_char is 'D' or 'S'
    """
    warp_positions = []

    # Only consider y rows 2-12 inclusive (zero indexed)
    min_y = 2
    max_y = 12

    for y in range(min_y, max_y + 1):
        for x in range(15):
            tile = traversability_map[y][x]
            if tile in {'D', 'S'}:
                warp_positions.append((x, y, tile))

    return warp_positions


def detect_door_exit_direction(
    traversability_map: List[List[str]],
    door_x: int,
    door_y: int
) -> Optional[Tuple[int, int]]:
    """
    Detect the exit direction for a door at (door_x, door_y).

    Priority:
    1. Check if door is part of a pair (adjacent 'D')
       - Horizontal pair: Exit perpendicular (up/down)
       - Vertical pair: Exit perpendicular (left/right)
    2. If single door, use pattern heuristics:
       - Prior traversable '.': +1 score
       - After blocked '#': +1 score
    3. Tie-breaker: South > North > West > East

    Args:
        traversability_map: 15x15 grid of traversability symbols
        door_x: X position of door (0-14)
        door_y: Y position of door (2-12)

    Returns:
        (dx, dy) tuple indicating exit direction, or None if cannot determine
    """
    # TODO: Verify assumption that doors always come in pairs

    # Step 1: Check if part of a pair (adjacent 'D')
    has_door_left = door_x > 0 and traversability_map[door_y][door_x - 1] == 'D'
    has_door_right = door_x < 14 and traversability_map[door_y][door_x + 1] == 'D'
    has_door_up = door_y > 2 and traversability_map[door_y - 1][door_x] == 'D'
    has_door_down = door_y < 12 and traversability_map[door_y + 1][door_x] == 'D'

    is_horizontal_pair = has_door_left or has_door_right
    is_vertical_pair = has_door_up or has_door_down

    candidates = []  # List of (direction_tuple, priority, score)

    if is_horizontal_pair:
        # Exit perpendicular to pair: up or down
        # Tie-breaker: South > North
        if door_y < 12:
            candidates.append(((0, 1), 0, 0))  # South, priority 0
        if door_y > 2:
            candidates.append(((0, -1), 1, 0))  # North, priority 1

    elif is_vertical_pair:
        # Exit perpendicular to pair: left or right
        # Tie-breaker: West > East
        if door_x > 0:
            candidates.append(((-1, 0), 2, 0))  # West, priority 2
        if door_x < 14:
            candidates.append(((1, 0), 3, 0))  # East, priority 3

    else:
        # Single door: use heuristics
        # Check all 4 directions for patterns

        # South (priority 0)
        if door_y < 12:
            score = _calculate_pattern_score(
                traversability_map, door_x, door_y, dx=0, dy=1
            )
            if score > 0:
                candidates.append(((0, 1), 0, score))

        # North (priority 1)
        if door_y > 2:
            score = _calculate_pattern_score(
                traversability_map, door_x, door_y, dx=0, dy=-1
            )
            if score > 0:
                candidates.append(((0, -1), 1, score))

        # West (priority 2)
        if door_x > 0:
            score = _calculate_pattern_score(
                traversability_map, door_x, door_y, dx=-1, dy=0
            )
            if score > 0:
                candidates.append(((-1, 0), 2, score))

        # East (priority 3)
        if door_x < 14:
            score = _calculate_pattern_score(
                traversability_map, door_x, door_y, dx=1, dy=0
            )
            if score > 0:
                candidates.append(((1, 0), 3, score))

    if not candidates:
        logger.warning(f"Could not detect exit direction for door at ({door_x}, {door_y})")
        return None

    # Sort by: score (higher better), then priority (lower better) as tie-breaker
    candidates.sort(key=lambda x: (-x[2], x[1]))

    direction = candidates[0][0]
    logger.debug(f"Door at ({door_x}, {door_y}) exit direction: {direction}")
    return direction


def _calculate_pattern_score(
    traversability_map: List[List[str]],
    door_x: int,
    door_y: int,
    dx: int,
    dy: int
) -> int:
    """
    Score direction based on ['.', 'D'] or ['D', '#'] patterns.

    Pattern: [prior, door, after] where we move in (dx, dy) direction
    - prior == '.': +1 (can approach from this side)
    - after == '#': +1 (blocked on other side)

    Args:
        traversability_map: 15x15 grid of traversability symbols
        door_x: X position of door
        door_y: Y position of door
        dx: X direction offset
        dy: Y direction offset

    Returns:
        Score for this direction (0-2)
    """
    score = 0

    # Prior tile (where we approach from)
    prior_x, prior_y = door_x - dx, door_y - dy
    if 0 <= prior_x < 15 and 2 <= prior_y <= 12:
        if traversability_map[prior_y][prior_x] == '.':
            score += 1

    # After tile (where we exit to)
    after_x, after_y = door_x + dx, door_y + dy
    if 0 <= after_x < 15 and 2 <= after_y <= 12:
        if traversability_map[after_y][after_x] == '#':
            score += 1

    return score


def prioritize_targets(targets: List[NavigationTarget]) -> List[NavigationTarget]:
    """
    Placeholder for future social reasoning and priority scoring.

    Args:
        targets: List of navigation targets

    Returns:
        Same list (currently unchanged, hook for future enhancements)
    """
    # TODO: this might involve reasoning or something, should this be in the agent class instead?
    return targets


def check_targets_reachability(
    targets: List[NavigationTarget],
    traversability_map: List[List[str]],
    player_local_tile_pos: Tuple[int, int] = (7, 7)
) -> List[NavigationTarget]:
    """
    Check reachability of navigation targets using A* pathfinding.

    For each target, attempts to find a path from the player position to the target.
    Updates the 'reachable' field of each target based on whether a path exists.

    Uses A* without fallback to ensure accurate reachability detection.

    Args:
        targets: List of navigation targets to check
        traversability_map: 15x15 grid of traversability symbols
        player_local_tile_pos: Player's local tile position (default is center at (7, 7))

    Returns:
        Same list of targets with 'reachable' field updated
    """
    # Import here to avoid circular dependency
    from custom_utils.navigation_astar import AStarPathfinder, TerrainGrid

    if not targets:
        return targets

    pathfinder = AStarPathfinder()
    terrain = TerrainGrid(traversability_map)

    # Build edge graph once for all targets (more efficient)
    edges = pathfinder._game_to_cost_grid_edges(terrain, obstacles=None)

    for target in targets:
        try:
            # Use A* directly without fallback for accurate reachability check
            path = pathfinder._run_astar_edges(
                start=player_local_tile_pos,
                goal=target.local_tile_position,
                edges=edges
            )

            # Target is reachable if A* finds a path (None means no path)
            target.reachable = (path is not None and len(path) > 0)

            if target.reachable:
                logger.debug(f"Target '{target.description}' is reachable (path length: {len(path)})")
            else:
                logger.debug(f"Target '{target.description}' is unreachable (no path found)")

        except Exception as e:
            # If pathfinding fails due to error, mark as unreachable
            logger.warning(f"Error checking reachability for target '{target.description}': {e}")
            target.reachable = False

    return targets


# Helper functions for target selection (shared by planner and navigation agent)

def format_targets_for_prompt(targets: List[NavigationTarget]) -> str:
    """
    Format navigation targets for VLM prompt.

    Shared utility for consistent target formatting across planner and navigation agent.
    Includes reachability information if available.

    Args:
        targets: List of navigation targets

    Returns:
        Formatted string with indexed target descriptions and reachability status
    """
    if not targets:
        return "No targets detected."

    lines = []
    for i, target in enumerate(targets):
        # Format with reachability status if available
        if target.reachable is not None:
            status = "[REACHABLE]" if target.reachable else "[UNREACHABLE]"
            lines.append(f"{i}. {status} {target.description}")
        else:
            # Fallback if reachability not checked
            lines.append(f"{i}. {target.description}")
    return "\n".join(lines)


def validate_and_select_target(
    target_index: int,
    targets: List[NavigationTarget]
) -> NavigationTarget:
    """
    Validate target index and return the selected target.

    Shared utility for consistent target validation across planner and navigation agent.
    Falls back to first target if index is invalid.

    Args:
        target_index: Index chosen by VLM (0-based)
        targets: List of available navigation targets

    Returns:
        Selected NavigationTarget (or first target as fallback)
    """
    if not targets:
        raise ValueError("Cannot select target from empty target list")

    if 0 <= target_index < len(targets):
        return targets[target_index]
    else:
        # Fallback to first target if index is out of bounds
        return targets[0]

