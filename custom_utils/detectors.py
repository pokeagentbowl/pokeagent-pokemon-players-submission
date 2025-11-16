"""
Game State Detectors

Utility functions for detecting various game states:
- Dialogue detection
- Player direction detection
- Player visibility detection
"""
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
import cv2

if TYPE_CHECKING:
    from utils.vlm import VLM

logger = logging.getLogger(__name__)


def detect_battle(frame: np.ndarray, threshold: float = 0.60) -> bool:
    """
    Detect if battle is initialized by checking for green pixels in bottom 1/4 of screen.
    
    Args:
        frame: Game frame as numpy array (480x352 or 240x160)
        threshold: Minimum white pixel ratio to detect dialogue (default 0.75)
        
    Returns:
        True if battle detected, False otherwise
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Ensure frame is upscaled to 480x320
        if frame_pil.size[1] == 160:
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
        
        # If frame is 352px height (480x352 with padding), crop out bottom 16px
        if frame_pil.size[1] == 352:
            frame_pil = frame_pil.crop((0, 16, frame_pil.size[0], 336))
        
        # Crop bottom 1/4 of screen
        width, height = frame_pil.size
        bottom_quarter = frame_pil.crop((0, (height * 3 // 4) -20, width, height))

        # # save bottom_quarter to debug folder
        # debug_path = "./debug_dialogue_bottom_quarter.png"
        # bottom_quarter.save(debug_path)
        
        # Convert to numpy and count white pixels
        bottom_array = np.array(bottom_quarter)
        
        # Check for pure white (255, 255, 255)
        green_mask = np.all(bottom_array == [107, 165, 165], axis=-1)
        green_ratio = np.sum(green_mask) / green_mask.size
        
        detected = green_ratio > threshold
        
        print(f"Battle detected: green_ratio={green_ratio:.3f}")
        
        return detected
        
    except Exception as e:
        logger.error(f"Failed to detect dialogue: {e}")
        return False

def detect_dialogue(frame: np.ndarray, threshold: float = 0.60) -> bool:
    """
    Detect if dialogue box is present by checking for white pixels in bottom 1/4 of screen.
    
    Args:
        frame: Game frame as numpy array (480x352 or 240x160)
        threshold: Minimum white pixel ratio to detect dialogue (default 0.75)
        
    Returns:
        True if dialogue detected, False otherwise
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Ensure frame is upscaled to 480x320
        if frame_pil.size[1] == 160:
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
        
        # If frame is 352px height (480x352 with padding), crop out bottom 16px
        if frame_pil.size[1] == 352:
            frame_pil = frame_pil.crop((0, 16, frame_pil.size[0], 336))
        
        # Crop bottom 1/4 of screen
        width, height = frame_pil.size
        bottom_quarter = frame_pil.crop((0, (height * 3 // 4) -20, width, height))

        # # save bottom_quarter to debug folder
        # debug_path = "./debug_dialogue_bottom_quarter.png"
        # bottom_quarter.save(debug_path)
        
        # Convert to numpy and count white pixels
        bottom_array = np.array(bottom_quarter)
        
        # Check for pure white (255, 255, 255)
        white_mask = np.all(bottom_array == [255, 255, 255], axis=-1)
        white_ratio = np.sum(white_mask) / white_mask.size
        
        detected = white_ratio > threshold
        
        print(f"Dialogue detected: white_ratio={white_ratio:.3f}")
        
        return detected
        
    except Exception as e:
        logger.error(f"Failed to detect dialogue: {e}")
        return False


def detect_player_direction(frame: np.ndarray, player_icons_dir: str = "startup_cache/playericons",
                           map_uid: Optional[str] = None, match_threshold: float = 0.8) -> Optional[str]:
    """
    Detect player facing direction using template matching.
    
    Args:
        frame: Game frame as numpy array
        player_icons_dir: Directory containing player icon templates
        map_uid: Optional map UID to narrow down icon search
        match_threshold: Minimum match score (default 0.8)
        
    Returns:
        Direction string ('North', 'South', 'East', 'West') or None if not detected
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # print(f"Input frame size: {frame_pil.size}")
        
        # Ensure frame is upscaled to 480x320
        if frame_pil.size[1] == 160:
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
            # print(f"Upscaled to 480x320")
        
        # Add 16px padding top and bottom
        if frame_pil.size[1] == 320:
            padded_height = frame_pil.height + 32
            padded_frame = Image.new('RGB', (frame_pil.width, padded_height), (0, 0, 0))
            padded_frame.paste(frame_pil, (0, 16))
            frame_pil = padded_frame
            # print(f"Added padding, new size: {frame_pil.size}")
        
        # Player is at tile (7, 5) in 0-indexed (x, y) coordinates
        # Frame is 480x352 = 15 tiles wide x 11 tiles tall (32x32 tiles)
        player_tile_x = 7  # Column 7 (0-indexed)
        player_tile_y = 5  # Row 5 (0-indexed)
        
        # Calculate pixel position
        # Each tile is 32x32, so player is at pixel (7*32, 5*32) = (224, 160)
        player_pixel_x = player_tile_x * 32
        player_pixel_y = player_tile_y * 32
        
        # print(f"Player crop position: ({player_pixel_x}, {player_pixel_y}) to ({player_pixel_x+32}, {player_pixel_y+32})")
        
        # Crop player tile (32x32)
        player_tile = frame_pil.crop((
            player_pixel_x,
            player_pixel_y,
            player_pixel_x + 32,
            player_pixel_y + 32
        ))
        
        player_tile_array = np.array(player_tile)
        # print(f"Player tile shape: {player_tile_array.shape}")
        
        # Load player icon templates
        icons_dir = Path(player_icons_dir)
        if not icons_dir.exists():
            logger.warning(f"Player icons directory not found: {icons_dir}")
            return None
        
        # If map_uid provided, check map-specific icons first
        search_dirs = []
        if map_uid:
            map_icons_dir = icons_dir / map_uid
            if map_icons_dir.exists():
                search_dirs.append(map_icons_dir)
        
        # Add generic icons directory and its subdirectories
        search_dirs.append(icons_dir)
        
        # print(f"Searching in {len(search_dirs)} directories for player templates")
        
        best_match_score = 0.0
        best_direction = None
        
        # Direction mapping from filename and directory
        direction_keywords = {
            'up': 'North',
            'north': 'North',
            'down': 'South',
            'south': 'South',
            'left': 'West',
            'west': 'West',
            'right': 'East',
            'east': 'East'
        }
        
        template_count = 0
        for search_dir in search_dirs:
            # Search recursively for all PNG files (including subdirectories)
            for icon_path in search_dir.rglob("*.png"):
                try:
                    icon_img = Image.open(icon_path).convert('RGB')
                    icon_array = np.array(icon_img)
                    
                    # Template matching
                    result = cv2.matchTemplate(
                        player_tile_array,
                        icon_array,
                        cv2.TM_CCOEFF_NORMED
                    )
                    
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    template_count += 1
                    
                    if max_val > best_match_score:
                        best_match_score = max_val
                        
                        # Extract direction from parent directory name or filename
                        parent_dir = icon_path.parent.name.lower()
                        filename = icon_path.stem.lower()
                        
                        # Check parent directory first (e.g., "up/", "down/")
                        for keyword, direction in direction_keywords.items():
                            if keyword in parent_dir or keyword in filename:
                                best_direction = direction
                                # print(f"New best match: {icon_path.relative_to(icons_dir)} = {max_val:.3f} ({direction})")
                                break
                
                except Exception as e:
                    print(f"Failed to match icon {icon_path}: {e}")
                    continue
        
        logger.info(f"Tested {template_count} templates, best score: {best_match_score:.3f}, direction: {best_direction}, threshold: {match_threshold}")
        
        if best_match_score >= match_threshold and best_direction:
            print(f"Player direction detected: {best_direction} (score={best_match_score:.3f})")
            return (best_direction, best_match_score)
        
        # Fallback: Try matching with top half of tile (16px) with lower threshold
        logger.info("Primary detection failed, trying fallback with top half of tile")
        
        # Crop top 16 pixels of the player tile
        player_tile_top_half = player_tile.crop((0, 0, 32, 16))
        player_tile_top_array = np.array(player_tile_top_half)
        
        fallback_threshold = 0.7
        fallback_best_score = 0.0
        fallback_best_direction = None
        
        fallback_template_count = 0
        for search_dir in search_dirs:
            for icon_path in search_dir.rglob("*.png"):
                try:
                    icon_img = Image.open(icon_path).convert('RGB')
                    
                    # Crop top 16 pixels of template
                    if icon_img.height >= 16:
                        icon_top_half = icon_img.crop((0, 0, icon_img.width, 16))
                        icon_top_array = np.array(icon_top_half)
                        
                        # Only match if dimensions are compatible
                        if icon_top_array.shape[0] == player_tile_top_array.shape[0] and \
                           icon_top_array.shape[1] == player_tile_top_array.shape[1]:
                            
                            result = cv2.matchTemplate(
                                player_tile_top_array,
                                icon_top_array,
                                cv2.TM_CCOEFF_NORMED
                            )
                            
                            _, max_val, _, _ = cv2.minMaxLoc(result)
                            fallback_template_count += 1
                            
                            if max_val > fallback_best_score:
                                fallback_best_score = max_val
                                
                                parent_dir = icon_path.parent.name.lower()
                                filename = icon_path.stem.lower()
                                
                                for keyword, direction in direction_keywords.items():
                                    if keyword in parent_dir or keyword in filename:
                                        fallback_best_direction = direction
                                        break
                
                except Exception as e:
                    continue
        
        logger.info(f"Fallback tested {fallback_template_count} templates, best score: {fallback_best_score:.3f}, direction: {fallback_best_direction}, threshold: {fallback_threshold}")
        
        if fallback_best_score >= fallback_threshold and fallback_best_direction:
            print(f"Player direction detected (fallback): {fallback_best_direction} (score={fallback_best_score:.3f})")
            return (fallback_best_direction, fallback_best_score)
        else:
            print(f"Player direction not detected (fallback also failed, best_score={fallback_best_score:.3f}, threshold={fallback_threshold})")
            # Return None if below threshold - caller should not proceed without confident detection
            return None
        
    except Exception as e:
        logger.error(f"Failed to detect player direction: {e}")
        return None


def detect_player_visible(frame: np.ndarray, player_icons_dir: str = "startup_cache/playericons",
                         map_uid: Optional[str] = None, match_threshold: float = 0.7) -> bool:
    """
    Detect if player is visible in frame (not in menu/battle).
    
    Args:
        frame: Game frame as numpy array
        player_icons_dir: Directory containing player icon templates
        map_uid: Optional map UID to narrow down icon search
        match_threshold: Minimum match score (default 0.7)
        
    Returns:
        True if player visible, False otherwise
    """
    direction_result = detect_player_direction(frame, player_icons_dir, map_uid, match_threshold)
    return direction_result is not None


def extract_dialogue_text_bbox(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box of dialogue text area.
    
    Args:
        frame: Game frame as numpy array
        
    Returns:
        Bounding box as (x1, y1, x2, y2) or None if not detected
    """
    try:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Ensure frame is upscaled to 480x320
        if frame_pil.size[1] == 160:
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
        
        width, height = frame_pil.size
        
        # Dialogue box is in bottom 1/4 of screen
        # Typical Pokemon dialogue box is bottom 80 pixels of 320px height
        dialogue_y1 = height * 3 // 4
        dialogue_y2 = height
        
        return (0, dialogue_y1, width, dialogue_y2)
        
    except Exception as e:
        logger.error(f"Failed to extract dialogue bbox: {e}")
        return None


def extract_dialogue_text(frame: np.ndarray, vlm: Optional['VLM'] = None) -> str:
    """
    Extract dialogue text from the frame using VLM.
    
    Args:
        frame: Game frame as numpy array
        vlm: VLM instance for text extraction (optional)
    
    Returns:
        Extracted dialogue text or error message
    """
    if vlm is None:
        logger.warning("VLM not available for dialogue text extraction")
        return "DIALOGUE_TEXT_PLACEHOLDER"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            frame_pil = Image.fromarray(frame)
        else:
            frame_pil = frame
        
        # Ensure frame is upscaled to 480x320
        if frame_pil.size[1] == 160:
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
        
        # Crop to bottom 25% of the screen where dialogue appears
        bottom_quarter_height = frame_pil.height // 4
        dialogue_region = frame_pil.crop((0, frame_pil.height - bottom_quarter_height, frame_pil.width, frame_pil.height))
        
        # Convert cropped PIL to numpy array for VLM
        frame_np = np.array(dialogue_region)
        
        prompt = """You are analyzing a Pokemon game screenshot. 

Look at the bottom portion of the screen where dialogue text appears. Extract and return ONLY the dialogue text that is currently visible. 

Rules:
- Read and return only the actual dialogue text content from the screenshot
- If no dialogue is visible, return "NO_DIALOGUE"
- Keep the text exactly as it appears in the game

Output format: Just the dialogue text, nothing else.
"""
        
        response = vlm.get_query(frame_np, prompt)
        
        print(f"Dialogue extraction response: {response}")
        
        # Clean up the response
        if response:
            # Remove any extra formatting or markers
            dialogue_text = response.strip()
            if dialogue_text.upper() in ["NO_DIALOGUE", "NO DIALOGUE", ""]:
                return "NO_DIALOGUE"
            return dialogue_text
        else:
            return "DIALOGUE_EXTRACTION_FAILED"
            
    except Exception as e:
        logger.error(f"Error extracting dialogue text with VLM: {e}")
        return "DIALOGUE_EXTRACTION_ERROR"
