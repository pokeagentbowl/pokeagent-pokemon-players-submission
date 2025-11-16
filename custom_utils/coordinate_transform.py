"""
Coordinate transformation utilities for pixel/local_tile/map_tile coordinate systems.

Coordinate Systems:
- Pixel: 240x160 frame (width x height) - screen space
- Local Tile: 15x15 player-centered tile grid with player always at center (7, 7)
  - Visible portion: 15x10 (width x height) maps to screen
  - Full grid: 15x15 includes tiles outside visible screen
- Map Tile: Absolute map coordinates from RAM

The pixel space (240x160) maps to a visible area of 15 wide x 10 tall.
The visible range spans local tile rows 2.5 to 11.5 (10 rows total), centered around player at row 7.
This gives 4.5 local tile cells above and 4.5 below the player:
- Upper 0.5 cell: Pixels 0-7 map to row 2 (partial visibility)
- Full cells: Pixels 8-151 map to rows 3-11
- Lower 0.5 cell: Pixels 152-159 map to row 11 (partial visibility)
- Player at row 7, which contains center pixel 80

Player is always at local tile center (7, 7) in the 15x15 grid.
"""
from typing import Tuple


# Constants
FRAME_WIDTH = 240
FRAME_HEIGHT = 160
LOCAL_TILE_WIDTH = 15
LOCAL_TILE_HEIGHT_VISIBLE = 10  # Visible height on screen
LOCAL_TILE_HEIGHT_FULL = 15     # Full local tile grid height
PLAYER_LOCAL_TILE_X = 7
PLAYER_LOCAL_TILE_Y = 7


def pixel_to_local_tile(
    pixel_x: int, 
    pixel_y: int,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT
) -> Tuple[int, int]:
    """
    Convert pixel coordinates to local tile position.
    
    Maps 240x160 pixel space to 15x10 visible local tiles, then places in 15x15 full local tile grid.
    The visible range spans from local tile row 2.5 to 11.5 (4.5 rows above and below player at row 7).
    
    Algorithm:
    - Direct mapping: local_tile_y = 2.5 + (pixel_y / 160) * 10
    - This places the 10 visible rows symmetrically around the player
    - The uppermost 0.5 and lowermost 0.5 represent partial cell visibility
    
    Args:
        pixel_x: X coordinate in pixel space (0-239)
        pixel_y: Y coordinate in pixel space (0-159)
        frame_width: Frame width in pixels (default: 240)
        frame_height: Frame height in pixels (default: 160)
        
    Returns:
        Tuple of (local_tile_x, local_tile_y) in 15x15 local tile space (0-14, 0-14)
    """
    # Direct mapping to achieve perfect centering with 4.5 cells above/below player
    # local_tile_y range: 2.5 + [0, 10) = [2.5, 12.5)
    # With int() truncation: [2, 12]
    local_tile_x_float = (pixel_x / frame_width) * LOCAL_TILE_WIDTH
    local_tile_y_float = 2.5 + (pixel_y / frame_height) * LOCAL_TILE_HEIGHT_VISIBLE
    
    # Convert to integer local tile positions
    local_tile_x = int(local_tile_x_float)
    local_tile_y = int(local_tile_y_float)
    
    # Clamp to valid local tile bounds
    # Y clamped to visible range [2, 12] - top/bottom 0.5 cells are at boundaries
    local_tile_x = max(0, min(LOCAL_TILE_WIDTH - 1, local_tile_x))
    local_tile_y = max(2, min(12, local_tile_y))
    
    return local_tile_x, local_tile_y


def local_tile_to_map_tile(
    local_tile_x: int,
    local_tile_y: int,
    player_map_tile_x: int,
    player_map_tile_y: int
) -> Tuple[int, int]:
    """
    Convert local tile coordinates to map tile coordinates.
    
    Player is at local tile center (7, 7), so we offset from player's map tile position.
    
    Algorithm:
        map_tile_x = player_map_tile_x + (local_tile_x - 7)
        map_tile_y = player_map_tile_y + (local_tile_y - 7)
    
    Args:
        local_tile_x: X position in local tile grid (0-14)
        local_tile_y: Y position in local tile grid (0-14)
        player_map_tile_x: Player's map tile X coordinate
        player_map_tile_y: Player's map tile Y coordinate
        
    Returns:
        Tuple of (map_tile_x, map_tile_y) in absolute map tile coordinates
    """
    map_tile_x = player_map_tile_x + (local_tile_x - PLAYER_LOCAL_TILE_X)
    map_tile_y = player_map_tile_y + (local_tile_y - PLAYER_LOCAL_TILE_Y)
    
    return map_tile_x, map_tile_y


def map_tile_to_local_tile(
    map_tile_x: int,
    map_tile_y: int,
    player_map_tile_x: int,
    player_map_tile_y: int
) -> Tuple[int, int]:
    """
    Convert map tile coordinates to local tile coordinates.
    
    Inverse of local_tile_to_map_tile.
    
    Algorithm:
        local_tile_x = map_tile_x - player_map_tile_x + 7
        local_tile_y = map_tile_y - player_map_tile_y + 7
    
    Args:
        map_tile_x: Map tile X coordinate
        map_tile_y: Map tile Y coordinate
        player_map_tile_x: Player's map tile X coordinate
        player_map_tile_y: Player's map tile Y coordinate
        
    Returns:
        Tuple of (local_tile_x, local_tile_y) in local tile space (may be outside 0-14 bounds)
    """
    local_tile_x = map_tile_x - player_map_tile_x + PLAYER_LOCAL_TILE_X
    local_tile_y = map_tile_y - player_map_tile_y + PLAYER_LOCAL_TILE_Y
    
    return local_tile_x, local_tile_y


def is_within_local_tile(local_tile_x: int, local_tile_y: int) -> bool:
    """
    Check if local tile position is within 15x15 bounds.
    
    Args:
        local_tile_x: X position in local tile grid
        local_tile_y: Y position in local tile grid
        
    Returns:
        True if within bounds (0-14, 0-14)
    """
    return 0 <= local_tile_x < LOCAL_TILE_WIDTH and 0 <= local_tile_y < LOCAL_TILE_HEIGHT_FULL


def is_within_visible_local_tile(local_tile_x: int, local_tile_y: int) -> bool:
    """
    Check if local tile position is within visible 15x10 area.
    
    Visible area is rows 2-12 (10 rows, centered around player at row 7).
    This corresponds to the visible range [2.5, 11.5) with 4.5 cells above/below player.
    
    Args:
        local_tile_x: X position in local tile grid
        local_tile_y: Y position in local tile grid
        
    Returns:
        True if within visible bounds
    """
    # Visible Y range: [2, 12] (inclusive)
    return 0 <= local_tile_x < LOCAL_TILE_WIDTH and 2 <= local_tile_y <= 12

