"""
Utility to extract clean 15x15 map grid from decorated visual_map.

The visual_map from map_stitcher includes decorations like:
- Header line ("--- MAP: LOCATION_NAME ---")
- Coordinate labels (X/Y axis)
- Footer with coordinate explanations
- Legend section

This utility extracts just the map tiles centered at the player (P).
"""
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_grid(grid_15x15: List[List[str]]) -> None:
    # Log the resultant grid
    grid_str = "\n".join("".join(row) for row in grid_15x15)
    # logger.info(f"Extracted grid:\n{grid_str}")
    
    # Validate dimensions
    if len(grid_15x15) != 15:
        raise ValueError(f"Grid height is {len(grid_15x15)}, expected 15.")

    for idx, row in enumerate(grid_15x15):
        if len(row) != 15:
            raise ValueError(f"Grid row {idx} has width {len(row)}, expected 15.")
    

def extract_15x15_grid_from_visual_map(
    visual_map: str,
    fallback_grid: Optional[List[List[str]]] = None
) -> List[List[str]]:
    """
    Extract a 15x15 grid centered at player from decorated visual_map.
    
    The visual_map is a formatted string with decorations. This function:
    1. Parses the visual_map to find map content lines
    2. Locates the player symbol 'P'
    3. Extracts a 15x15 grid centered at the player
    4. Returns as a 2D list of single-character strings
    
    Args:
        visual_map: Decorated map string from map_stitcher
        fallback_grid: Fallback grid to use if extraction fails (default: 15x15 of '.')
        
    Returns:
        15x15 grid as List[List[str]], centered at player if found
    """
    if fallback_grid is None:
        fallback_grid = [['.' for _ in range(15)] for _ in range(15)]
    
    if not visual_map:
        logger.warning("Empty visual_map, returning fallback grid")
        return fallback_grid
    
    try:
        # Split into lines
        lines = visual_map.split('\n')
        
        # Find map content lines (exclude header, footer, legend)
        # Map content lines have Y-axis coordinate labels or are pure map rows
        map_content_lines = []
        in_map_section = False
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Skip header line (starts with "---")
            if line.strip().startswith("---"):
                in_map_section = True
                continue
            
            # Stop at coordinate explanation or legend
            if "Player at" in line or "Legend:" in line or "Map shows GAME" in line or "Movement:" in line:
                break
            
            # Skip X-axis coordinate label line (all digits/spaces at start of map section)
            if in_map_section and not map_content_lines:
                # Check if line looks like coordinate labels (starts with spaces followed by numbers)
                stripped = line.strip()
                if stripped and all(c.isdigit() or c.isspace() for c in stripped):
                    continue
            
            # Map content line
            if in_map_section:
                map_content_lines.append(line)
        
        if not map_content_lines:
            logger.warning("No map content lines found in visual_map")
            return fallback_grid
        
        # Known tile symbols from map_stitcher.py (excluding space)
        # Space is excluded because it can be both a tile AND a separator
        # We use non-space symbols as keystones to determine alignment
        TILE_SYMBOLS = {
            '.', '#', '~', '^', 'W', 'I', 's', 'D', 'S', 'C',
            '→', '←', '↑', '↓', '↗', '↖', '↘', '↙', 'L', 'T',
            'G', 'K', 'B', '?', 'P', 'N'
        }
        
        # Extract map grid from lines (remove Y-axis labels and spacing)
        raw_grid = []
        for line in map_content_lines:
            # Find where map content starts by detecting first tile symbol
            # Y-axis label format: "YYY  " (3 digits + 2 spaces)
            # But coordinates can be negative: " -4  " or positive: "  0  "
            
            content_start = 0
            if len(line) > 5:
                # Check if line starts with coordinate (digits, spaces, or minus)
                prefix = line[:5].strip()
                if prefix.lstrip('-').isdigit():
                    # Line has Y-axis label - Y-label is ALWAYS exactly 5 chars
                    # Format: f"{coord:3d}  " (3 digits right-aligned + 2 spaces)
                    # Reason: Can't search for first tile symbol because first tile might BE a space
                    content_start = 5
            
            map_content = line[content_start:]
            
            if not map_content:
                continue
            
            # Extract tiles at positions 0, 2, 4, ... (tiles are separated by single spaces)
            # Use TILE_SYMBOLS (non-space symbols) as keystones to determine alignment
            # Reason: Space can be a tile OR separator, so we anchor on definite tile symbols
            
            # Find first non-space tile symbol as keystone
            keystone_pos = None
            for i in range(len(map_content)):
                if map_content[i] in TILE_SYMBOLS:
                    keystone_pos = i
                    break
            
            if keystone_pos is None:
                # No keystone found - line might be all spaces
                # Default to even positions (0, 2, 4, ...)
                tiles = [map_content[i] for i in range(0, len(map_content), 2)]
            elif keystone_pos % 2 == 0:
                # Keystone at even position - tiles are at 0, 2, 4, ...
                tiles = [map_content[i] for i in range(0, len(map_content), 2)]
            else:
                # Keystone at odd position - tiles are at 1, 3, 5, ...
                tiles = [map_content[i] for i in range(1, len(map_content), 2)]
            
            if tiles:
                raw_grid.append(tiles)
        
        if not raw_grid:
            logger.warning("Failed to extract tiles from map content")
            return fallback_grid
        
        # Find player position in raw_grid
        player_pos = None
        for y, row in enumerate(raw_grid):
            for x, tile in enumerate(row):
                if tile == 'P':
                    player_pos = (x, y)
                    break
            if player_pos:
                break
        
        if not player_pos:
            logger.warning("Player symbol 'P' not found in visual_map, using center of grid")
            # Use center of raw_grid as player position
            if raw_grid:
                player_pos = (len(raw_grid[0]) // 2, len(raw_grid) // 2)
            else:
                return fallback_grid
        
        # Extract 15x15 grid centered at player
        player_x, player_y = player_pos
        grid_15x15 = []
        
        for dy in range(-7, 8):  # -7 to 7 inclusive (15 rows)
            row_15 = []
            for dx in range(-7, 8):  # -7 to 7 inclusive (15 cols)
                src_x = player_x + dx
                src_y = player_y + dy
                
                # Check bounds in raw_grid
                if 0 <= src_y < len(raw_grid) and 0 <= src_x < len(raw_grid[src_y]):
                    tile = raw_grid[src_y][src_x]
                else:
                    tile = '?'  # Unknown/unexplored
                
                row_15.append(tile)
            
            grid_15x15.append(row_15)
        
        validate_grid(grid_15x15)
        return grid_15x15
        
    except Exception as e:
        logger.error(f"Failed to extract 15x15 grid from visual_map: {e}")
        return fallback_grid


def get_player_centered_grid(
    map_data: dict,
    fallback_grid: Optional[List[List[str]]] = None
) -> List[List[str]]:
    """
    Convenience function to extract grid from map_data dict.
    
    Args:
        map_data: Map data dict with 'visual_map' key
        fallback_grid: Fallback grid if extraction fails
        
    Returns:
        15x15 grid centered at player
    """
    visual_map = map_data.get('visual_map', '')
    return extract_15x15_grid_from_visual_map(visual_map, fallback_grid)

