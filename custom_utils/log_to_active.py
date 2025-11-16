"""
Caching utilities for overall agent.

Handles loading/saving grass cache and actions log.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from custom_utils.label_traversable import compute_simple_tile_features

logger = logging.getLogger(__name__)

# Constants
ACTIONS_LOG_FILE = "./startup_cache/actions.json"
GRASS_CACHE_FILE = "./startup_cache/grass_cache.json"
NAVIGATION_CACHE_DIR = "./navigation_caches"
TARGET_SELECTION_CACHE = Path(NAVIGATION_CACHE_DIR) / "target_selection_counts.json"
INTERACTION_LOG_FILE = Path(NAVIGATION_CACHE_DIR) / "interactions.json"
PORTAL_CONNECTIONS_CACHE = Path(NAVIGATION_CACHE_DIR) / "portal_connections.json"


def load_grass_cache() -> Dict[str, Dict[Tuple[int, int], int]]:
    """
    Load grass cache from JSON file.

    Format: {"map_name": {"x,y": last_visited_step_num}}

    Returns:
        Dict mapping map_name to dict of (x,y) positions and last visited step
    """
    cache_path = Path(GRASS_CACHE_FILE)

    if not cache_path.exists():
        logger.info("No grass cache found, creating new cache")
        return {}

    try:
        with open(cache_path, 'r') as f:
            raw_cache = json.load(f)

        # Convert string keys back to tuples
        grass_cache = {}
        for map_name, positions in raw_cache.items():
            grass_cache[map_name] = {}
            for pos_str, step_num in positions.items():
                x, y = map(int, pos_str.split(','))
                grass_cache[map_name][(x, y)] = step_num

        logger.info(f"Loaded grass cache with {len(grass_cache)} maps")
        return grass_cache

    except Exception as e:
        logger.error(f"Failed to load grass cache: {e}")
        return {}


def save_grass_cache(grass_cache: Dict[str, Dict[Tuple[int, int], int]]) -> None:
    """Save grass cache to JSON file"""
    try:
        # Convert tuple keys to strings for JSON serialization
        raw_cache = {}
        for map_name, positions in grass_cache.items():
            raw_cache[map_name] = {}
            for pos, step_num in positions.items():
                pos_str = f"{pos[0]},{pos[1]}"
                raw_cache[map_name][pos_str] = step_num

        cache_path = Path(GRASS_CACHE_FILE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, 'w') as f:
            json.dump(raw_cache, f, indent=2)

        logger.debug(f"Saved grass cache with {len(grass_cache)} maps")

    except Exception as e:
        logger.error(f"Failed to save grass cache: {e}")


def load_actions_log() -> List[Dict]:
    """
    Load actions log from JSON file.

    Returns:
        List of action entries
    """
    log_path = Path(ACTIONS_LOG_FILE)

    if not log_path.exists():
        logger.info("No actions log found, creating new log")
        return []

    try:
        with open(log_path, 'r') as f:
            actions_log = json.load(f)

        logger.info(f"Loaded actions log with {len(actions_log)} entries")
        return actions_log

    except Exception as e:
        logger.error(f"Failed to load actions log: {e}")
        return []


def save_actions_log(actions_log: List[Dict]) -> None:
    """Save actions log to JSON file"""
    try:
        log_path = Path(ACTIONS_LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w') as f:
            json.dump(actions_log, f, indent=2)

        logger.debug(f"Saved actions log with {len(actions_log)} entries")

    except Exception as e:
        logger.error(f"Failed to save actions log: {e}")


def log_action(actions_log: List[Dict], step: int, reasoning: str, actions: List[str]) -> None:
    """
    Log an action to the actions log.

    Args:
        actions_log: The actions log list to update
        step: Step number
        reasoning: Reasoning for the action
        actions: List of actions taken
    """
    entry = {
        "step": step,
        "reasoning": reasoning,
        "actions": actions
    }
    actions_log.append(entry)
    save_actions_log(actions_log)


def update_grass_cache(grass_cache: Dict[str, Dict[Tuple[int, int], int]], map_name: str, position: Tuple[int, int], current_step_num: int) -> None:
    """
    Update grass cache with current position and step number.

    Args:
        grass_cache: The grass cache dict to update
        map_name: Current map name
        position: (x, y) position tuple
        current_step_num: Current step number
    """
    if map_name not in grass_cache:
        grass_cache[map_name] = {}

    grass_cache[map_name][position] = current_step_num
    save_grass_cache(grass_cache)


def ensure_cache_directories() -> None:
    """Ensure navigation cache directories exist."""
    cache_dir = Path(NAVIGATION_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create interaction logs directory
    interaction_dir = Path(NAVIGATION_CACHE_DIR) / "interactions.json"
    interaction_dir.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cache directories initialized: {NAVIGATION_CACHE_DIR}")


def load_active_tile_index() -> Dict[str, Dict]:
    """Load active tile index from cache."""
    index_path = Path(NAVIGATION_CACHE_DIR) / "active_tile_index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                data = json.load(f)
                
            # Handle both old list format and new dict format
            active_tile_index = {}
            if isinstance(data, list):
                # Old format: convert list to dict
                for item in data:
                    if isinstance(item, dict):
                        filename = item.get('filename')
                        tile_class = item.get('class', 'npc' if item.get('is_npc', False) else 'untraversable')
                        if filename:
                            active_tile_index[filename] = {
                                'class': tile_class,
                                'tile_pos': item.get('tile_pos', [0, 0])
                            }
                    elif isinstance(item, list) and len(item) == 2:
                        # Very old format: [filename, class]
                        active_tile_index[item[0]] = {
                            'class': item[1],
                            'tile_pos': [0, 0]
                        }
            else:
                # New format: already a dict, but validate structure
                for filename, tile_data in data.items():
                    # Ensure tile_data is in correct format: {'class': str, 'tile_pos': [x, y]}
                    if isinstance(tile_data, dict):
                        # Already in correct format
                        active_tile_index[filename] = {
                            'class': tile_data.get('class', 'untraversable'),
                            'tile_pos': tile_data.get('tile_pos', [0, 0])
                        }
                    elif isinstance(tile_data, list) and len(tile_data) >= 2:
                        # Old format: [class, [x, y]] or similar
                        tile_class = tile_data[0] if isinstance(tile_data[0], str) else 'untraversable'
                        tile_pos = tile_data[1] if isinstance(tile_data[1], list) and len(tile_data[1]) == 2 else [0, 0]
                        active_tile_index[filename] = {
                            'class': tile_class,
                            'tile_pos': tile_pos
                        }
                    else:
                        # Unknown format, skip
                        logger.warning(f"Skipping tile {filename} with unknown format: {type(tile_data)}")
                
            logger.info(f"Loaded active tile index: {len(active_tile_index)} tiles")
            return active_tile_index
        except Exception as e:
            logger.warning(f"Failed to load active tile index: {e}")
            return {}
    else:
        logger.info("No existing active tile index found")
        return {}


def save_active_tile_index(active_tile_index: Dict[str, Dict]) -> None:
    """Save active tile index to cache."""
    index_path = Path(NAVIGATION_CACHE_DIR) / "active_tile_index.json"
    try:
        with open(index_path, 'w') as f:
            json.dump(active_tile_index, f, indent=2)
        logger.debug(f"Saved active tile index: {len(active_tile_index)} tiles")
    except Exception as e:
        logger.error(f"Failed to save active tile index: {e}")


def load_target_selection_counts() -> Dict[str, int]:
    """Load target selection counts from cache."""
    if TARGET_SELECTION_CACHE.exists():
        try:
            with open(TARGET_SELECTION_CACHE, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded target selection counts: {len(data)} targets")
            return data
        except Exception as e:
            logger.warning(f"Failed to load target selection counts: {e}")
            return {}
    else:
        logger.info("No existing target selection counts found")
        return {}


def save_target_selection_counts(target_selection_counts: Dict[str, int]) -> None:
    """Save target selection counts to cache."""
    try:
        with open(TARGET_SELECTION_CACHE, 'w') as f:
            json.dump(target_selection_counts, f, indent=2)
        logger.debug(f"Saved target selection counts: {len(target_selection_counts)} targets")
    except Exception as e:
        logger.error(f"Failed to save target selection counts: {e}")


def log_interaction(
    action: str, 
    tile_pos: tuple, 
    result: str, 
    is_npc: bool = False,
    map_location: str = None,
    navigation_target: dict = None
) -> None:
    """
    Log interaction to interactions.json.
    
    Args:
        action: Action taken (e.g., 'A')
        tile_pos: (x, y) tile position
        result: Result of interaction
        is_npc: Whether this was an NPC interaction
        map_location: Current map location
        navigation_target: Current navigation target info
    """
    try:
        log_path = Path(INTERACTION_LOG_FILE)
        
        # Load existing log
        if log_path.exists():
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Add new entry
        log_data.append({
            'action': action,
            'tile_pos': tile_pos,
            'map_location': map_location,
            'result': result,
            'is_npc': is_npc,
            'navigation_target': navigation_target,
            'timestamp': str(Path(INTERACTION_LOG_FILE).stat().st_mtime if log_path.exists() else 0)
        })
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")


def save_tile_to_cache(
    tile_pos: tuple, 
    frame: np.ndarray, 
    filename: str, 
    tile_class: str,
    player_map_tile_x: int,
    player_map_tile_y: int
) -> None:
    """
    Crop and save a tile to the navigation cache.
    
    Args:
        tile_pos: (x, y) map tile position
        frame: Current game frame (should be 480x352 padded)
        filename: Filename to save as
        tile_class: Tile class ('npc', 'untraversable', 'grass')
        player_map_tile_x: Player's current map x position
        player_map_tile_y: Player's current map y position
    """
    try:
        # Convert frame to PIL Image if needed
        try:
            if isinstance(frame, np.ndarray):
                frame_pil = Image.fromarray(frame.astype('uint8'), 'RGB')
            else:
                frame_pil = frame
        except Exception as e:
            logger.error(f"Failed to convert frame to PIL Image: {e}")
            raise
        
        # Preprocess frame: ensure it's 480x352 (upscale and pad if needed)
        try:
            if frame_pil.size[1] == 160:
                # 240x160 -> 480x320 (2x upscale)
                frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
                logger.debug(f"Upscaled frame from 240x160 to 480x320")
            
            if frame_pil.size[1] == 320:
                # 480x320 -> 480x352 (add 16px top and bottom padding)
                padded_frame = Image.new('RGB', (480, 352), (0, 0, 0))
                padded_frame.paste(frame_pil, (0, 16))
                frame_pil = padded_frame
                logger.debug(f"Padded frame from 480x320 to 480x352")
            
            # Verify final size
            if frame_pil.size != (480, 352):
                logger.warning(f"Unexpected frame size after preprocessing: {frame_pil.size}, expected (480, 352)")
        except Exception as e:
            logger.error(f"Failed to preprocess frame size: {e}")
            logger.error(f"Frame size: {frame_pil.size}")
            raise
        
        # Calculate tile position in frame
        # The 352px frame has already been processed so row 0 is a full 32x32 tile
        # Row 0 starts at pixel 0, row 1 at pixel 32, etc.
        # Player is at frame tile (7, 5) in 0-indexed coordinates
        # Column 7 = pixel x 224, Row 5 = pixel y 160
        
        tile_map_x, tile_map_y = tile_pos
        
        # Player is always at frame tile (7, 5) in 0-indexed coordinates
        # This means column 7 (pixel x = 7*32 = 224)
        # and row 5 (pixel y = 5*32 = 160)
        player_frame_x = 7
        player_frame_y = 5
        
        # Calculate frame tile position by offset from player
        frame_tile_x = player_frame_x + (tile_map_x - player_map_tile_x)
        frame_tile_y = player_frame_y + (tile_map_y - player_map_tile_y)
        
        logger.debug(f"Tile map pos ({tile_map_x}, {tile_map_y}), player map pos ({player_map_tile_x}, {player_map_tile_y}), frame tile pos ({frame_tile_x}, {frame_tile_y})")
        
        # Check if tile is in visible frame (columns 0-14, rows 0-10)
        if frame_tile_x < 0 or frame_tile_x >= 15 or frame_tile_y < 0 or frame_tile_y >= 11:
            logger.warning(f"Tile map pos ({tile_map_x}, {tile_map_y}) -> frame pos ({frame_tile_x}, {frame_tile_y}) is outside visible frame")
            return
        
        # Crop tile (32x32 pixels)
        # After preprocessing, the 480x352 frame has structure:
        # - Rows 0: Top padding (16px black)
        # - Rows 16-335: Game content (320px = 10 rows of 32px tiles)
        # - Rows 336-351: Bottom padding (16px black)
        # Frame tiles are indexed 0-10 (11 rows), with player at row 5
        # Row 0 starts at pixel 0, row 1 at pixel 32, row 5 at pixel 160, etc.
        try:
            left = frame_tile_x * 32
            top = frame_tile_y * 32  # No offset needed - tiles indexed from padded frame top
            right = left + 32
            bottom = top + 32
            
            tile_img = frame_pil.crop((left, top, right, bottom))
        except Exception as e:
            logger.error(f"Failed to crop tile at frame pos ({frame_tile_x}, {frame_tile_y}): {e}")
            raise
        
        # Compute features
        try:
            features = compute_simple_tile_features(tile_img)
        except Exception as e:
            logger.error(f"Failed to compute tile features for {filename}: {e}")
            raise
        
        # Save tile image to PNG
        try:
            cache_dir = Path(NAVIGATION_CACHE_DIR)
            tile_path = cache_dir / filename
            tile_img.save(tile_path)
            logger.debug(f"Saved tile image to {tile_path}")
        except Exception as e:
            logger.error(f"Failed to save tile image {filename}: {e}")
            raise
        
        # Load or create features file (flat key:value format)
        try:
            features_path = cache_dir / "active_tile_simplefeatures.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    features_data = json.load(f)
                
                # Validate that features_data is a dict, not a list
                if not isinstance(features_data, dict):
                    logger.warning(f"active_tile_simplefeatures.json contains {type(features_data)} instead of dict, resetting to empty dict")
                    features_data = {}
            else:
                features_data = {}
        except Exception as e:
            logger.error(f"Failed to load features file: {e}")
            raise
        
        # Add/update tile features (flat format: "filename": [features])
        try:
            features_data[filename] = features.tolist()
        except Exception as e:
            logger.error(f"Failed to add features to features_data for {filename}: {e}")
            logger.error(f"features type: {type(features)}, features_data type: {type(features_data)}")
            raise
        
        # Save features file
        try:
            with open(features_path, 'w') as f:
                json.dump(features_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save features file: {e}")
            raise
        
        logger.info(f"Saved tile {filename} to cache as {tile_class}")
        
    except Exception as e:
        import traceback
        logger.error(f"Failed to save tile to cache - filename: {filename}, tile_pos: {tile_pos}, tile_class: {tile_class}")
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


def mark_and_save_tile(
    tile_pos: tuple, 
    frame: np.ndarray, 
    tile_class: str, 
    active_tile_index: dict,
    map_location: Optional[str] = None, 
    allow_upgrade: bool = False,
    player_map_tile_x: int = None,
    player_map_tile_y: int = None
) -> None:
    """
    Unified function to mark and save a tile to the navigation cache.
    
    Handles detection, marking, and saving for all tile classes: 'npc', 'untraversable', 'grass'.
    
    Args:
        tile_pos: (x, y) map tile position
        frame: Current game frame
        tile_class: Tile class ('npc', 'untraversable', 'grass')
        active_tile_index: Active tile index dict to update
        map_location: Map location string
        allow_upgrade: If True, allows upgrading tiles to higher priority classes
                      (e.g., untraversable -> npc). Default False.
        player_map_tile_x: Player's current map x position
        player_map_tile_y: Player's current map y position
    """
    tile_x, tile_y = tile_pos
    
    # Use provided map_location or default to current player map
    if map_location is None:
        map_location = "unknown"
    
    # Format: {map_id}_{x}_{y}.png
    filename = f"{map_location}_{tile_x}_{tile_y}.png"
    
    # Check if already cached with same class
    if filename in active_tile_index:
        existing_class = active_tile_index[filename].get('class')
        
        if existing_class == tile_class:
            logger.debug(f"Tile {filename} already marked as {tile_class}")
            return
        
        # Handle upgrades if allowed
        if allow_upgrade and existing_class == 'untraversable' and tile_class == 'npc':
            logger.info(f"Upgrading tile {filename} from untraversable to NPC")
        elif not allow_upgrade and existing_class != tile_class:
            logger.warning(f"Tile {filename} already exists as {existing_class}, not overwriting with {tile_class}")
            return
    
    # Crop and save tile (this will update both features and index)
    save_tile_to_cache(
        tile_pos, 
        frame, 
        filename, 
        tile_class=tile_class,
        player_map_tile_x=player_map_tile_x,
        player_map_tile_y=player_map_tile_y
    )
    
    # Update active tile index
    active_tile_index[filename] = {
        'class': tile_class,
        'tile_pos': list(tile_pos)
    }
    
    # Save active tile index to file
    save_active_tile_index(active_tile_index)
    
    logger.info(f"Marked tile ({tile_x}, {tile_y}) as {tile_class}")


def load_portal_connections() -> Dict[str, List[Dict]]:
    """Load portal connections from cache."""
    if PORTAL_CONNECTIONS_CACHE.exists():
        try:
            with open(PORTAL_CONNECTIONS_CACHE, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded portal connections for {len(data)} maps")
            return data
        except Exception as e:
            logger.warning(f"Failed to load portal connections: {e}")
            return {}
    else:
        logger.info("No existing portal connections cache found")
        return {}


def save_portal_connections(portal_connections: Dict[str, List[Dict]]) -> None:
    """Save portal connections to cache."""
    try:
        with open(PORTAL_CONNECTIONS_CACHE, 'w') as f:
            json.dump(portal_connections, f, indent=2)
        logger.debug(f"Saved portal connections for {len(portal_connections)} maps")
    except Exception as e:
        logger.error(f"Failed to save portal connections: {e}")


def update_portal_connections_cache(map_name: str, connections: List[Dict]) -> None:
    """
    Update the portal connections cache for a specific map.
    
    Args:
        map_name: Name of the map
        connections: List of connection dicts for this map
    """
    logger.info(f"Updating portal connections cache for {map_name} with {len(connections)} connections")
    try:
        # Load existing cache
        cache = load_portal_connections()
        
        # Update with new connections
        cache[map_name] = connections
        
        # Save back to cache
        save_portal_connections(cache)
        
        logger.info(f"Updated portal connections cache for {map_name} with {len(connections)} connections")
        
    except Exception as e:
        logger.error(f"Failed to update portal connections cache for {map_name}: {e}")


def load_completed_dialogues() -> List[Dict]:
    """
    Load the last 5 completed dialogues from the dialogue log.

    Returns the completed dialogues with step, map, coordinate, text.
    Groups dialogues by dialogue_count to find complete sessions.

    Returns:
        List of completed dialogue dicts with step, map, coordinate, text
    """
    # Dialogue log file path (same as used in overall_agent_nt)
    DIALOGUE_LOG_FILE = "./startup_cache/dialogues.json"
    
    try:
        log_path = Path(DIALOGUE_LOG_FILE)

        # Load existing log
        if not log_path.exists():
            return []

        with open(log_path, 'r') as f:
            log_data = json.load(f)

        if not log_data:
            return []

        # Group dialogues by dialogue_count to find complete sessions
        dialogue_sessions = {}
        for dialogue in log_data:
            count = dialogue.get('dialogue_count')
            if count is None:
                continue

            if count not in dialogue_sessions:
                dialogue_sessions[count] = {'start': None, 'end': None}

            if dialogue.get('event') == 'dialogue_start':
                dialogue_sessions[count]['start'] = dialogue
            elif dialogue.get('event') == 'dialogue_end':
                dialogue_sessions[count]['end'] = dialogue

        # Extract completed dialogues (have both start and end)
        completed = []
        for count, session in dialogue_sessions.items():
            start_entry = session.get('start')
            end_entry = session.get('end')

            if start_entry is not None and end_entry is not None:
                # Get the dialogue text (prefer the end entry if it has text, otherwise start)
                dialogue_text = ""
                if end_entry.get('dialogue_text'):
                    dialogue_text = end_entry['dialogue_text']
                elif start_entry.get('dialogue_text'):
                    dialogue_text = start_entry['dialogue_text']

                if dialogue_text and dialogue_text not in ["EXTRACTION_FAILED", "DIALOGUE_TEXT_PLACEHOLDER", "NO_DIALOGUE"]:
                    coord = start_entry.get('player_position', {})
                    completed.append({
                        'step': start_entry.get('step_number', 0),
                        'map': start_entry.get('player_map', 'unknown'),
                        'coordinate': (coord.get('x', 0), coord.get('y', 0)),
                        'text': dialogue_text
                    })

        if not completed:
            return []

        # Sort by step number (most recent first) and take last 5
        completed.sort(key=lambda x: x['step'], reverse=True)
        recent_completed = completed[:5]

        # Deduplicate by text content (keep the most recent occurrence of each unique text)
        seen_texts = set()
        deduplicated = []
        for dialogue in recent_completed:
            text = dialogue['text']
            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated.append(dialogue)

        return deduplicated

    except Exception as e:
        logger.error(f"Failed to load completed dialogues: {e}")
        return []
