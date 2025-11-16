"""
Coordinate transformation utilities for pixel/local_tile/map_tile coordinate systems.

Coordinate Systems:
- Pixel/Frame: 480x352 frame (width x height) - padded frame space with player at (7,5) when present
- Local Tile: 15x15 player-centered tile grid with player always at center (7, 7)
  - Visible portion: 15x15 (width x height) maps to padded screen
- Map Tile: Absolute map coordinates from RAM

The pixel space (480x352) maps to a visible area of 15 wide x 11 tall.
The local tile maps to an area 15 x 15 and player is at local tile center (7, 7).
"""
from typing import Tuple


# Constants
FRAME_WIDTH = 480
FRAME_HEIGHT = 352  # Updated for padded frame (320 + 32)
LOCAL_TILE_WIDTH = 15
LOCAL_TILE_HEIGHT_VISIBLE = 15  # Updated for padded frame (352/32)
LOCAL_TILE_HEIGHT_FULL = 15     # Updated to match visible
PLAYER_LOCAL_TILE_X = 7
PLAYER_LOCAL_TILE_Y = 7  # Updated to match label_traversable player position


def pixel_to_local_tile(
    pixel_x: int, 
    pixel_y: int,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT
) -> Tuple[int, int]:
    """
    Convert pixel coordinates to local tile position.
    
    Maps 480x352 pixel space (padded frame) to 15x11 visible local tiles, centered with player at (7, 7).
    
    The frame is padded with 16px top and bottom (original 480x320 -> 480x352).
    Player is at frame tile (7, 5) which maps to local tile (7, 7).
    
    Algorithm:
    - Remove 16px top padding before conversion
    - Convert to frame tile coordinates (32x32 tiles)
    - Offset from player frame tile (7, 5) to get local tile
    
    Args:
        pixel_x: X coordinate in pixel space (0-479)
        pixel_y: Y coordinate in pixel space (0-351)
        frame_width: Frame width in pixels (default: 480)
        frame_height: Frame height in pixels (default: 352)
        
    Returns:
        Tuple of (local_tile_x, local_tile_y) in 15x15 local tile space (0-14, 0-14)
    """
    # Remove top padding (16px) from pixel_y
    adjusted_pixel_y = pixel_y - 16
    
    # Convert to frame tile coordinates (32x32 tiles)
    frame_tile_x = pixel_x // 32
    frame_tile_y = adjusted_pixel_y // 32
    
    # Player is at frame tile (7, 5), which maps to local tile (7, 7)
    # Calculate offset from player and add to player's local tile position
    local_tile_x = 7 + (frame_tile_x - 7)
    local_tile_y = 7 + (frame_tile_y - 5)
    
    # Clamp to valid local tile bounds
    local_tile_x = max(0, min(LOCAL_TILE_WIDTH - 1, local_tile_x))
    local_tile_y = max(0, min(LOCAL_TILE_HEIGHT_FULL - 1, local_tile_y))
    
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
    Check if local tile position is within visible 15x11 area.
    
    Visible area is rows 0-10 (11 rows, centered around player at row 5).
    
    Args:
        local_tile_x: X position in local tile grid
        local_tile_y: Y position in local tile grid
        
    Returns:
        True if within visible bounds (0-14, 0-10)
    """
    return 0 <= local_tile_x < LOCAL_TILE_WIDTH and 0 <= local_tile_y < LOCAL_TILE_HEIGHT_VISIBLE

