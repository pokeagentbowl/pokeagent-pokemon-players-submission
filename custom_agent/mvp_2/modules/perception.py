"""Perception module - object detection, scene understanding, and navigation targets.

Heavily modeled after custom_agent/navigation_agent.py with adaptations for hierarchical agent.
"""

import numpy as np
import logging
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from custom_utils.azure_vision_analyzer import DetectedObject, OCRTextBlock

from custom_utils.cached_azure_vision_analyzer import CachedAzureVisionAnalyzer
from custom_utils.map_extractor import get_player_centered_grid
from custom_utils.detectors import detect_dialogue, detect_player_visible
from custom_utils.navigation_targets_nt import NavigationTarget, generate_navigation_targets

# NT: Import object detector for NPC detection
from custom_utils.cached_object_detector_nt import CachedObjectDetector as ObjectDetector

# NT: Import log_to_active for cache functions
from custom_utils.log_to_active import (
    ensure_cache_directories, load_active_tile_index, save_active_tile_index,
    load_target_selection_counts, save_target_selection_counts,
    log_interaction, save_tile_to_cache, mark_and_save_tile,
    load_portal_connections, update_portal_connections_cache, INTERACTION_LOG_FILE
)

logger = logging.getLogger(__name__)
USEPERCEPTIONOVERRIDE = True
ensure_cache_directories()

class PerceptionResult(BaseModel):
    """Result of perception processing.

    Note: Embeddings are handled by the memory module, not perception.
    """
    detected_objects: List[DetectedObject]
    ocr_text: List[OCRTextBlock]  # OCR text extracted from frame
    scene_description: str
    navigation_targets: List[NavigationTarget]  # Available navigation targets
    llm_outputs: Optional[Dict[str, str]] = None

    class Config:
        arbitrary_types_allowed = True


class PerceptionModule:
    """
    Processes visual and game state into structured perception.
    Includes navigation target generation.

    Note:
    - scene_description is blank for MVP as per requirements
    - Embeddings are now handled by the memory module, not perception
    """
    
    def __init__(self, reasoner=None):
        """
        Initialize perception module.

        Args:
            reasoner: LangChainVLM instance (not used for MVP, kept for future compatibility)
        """
        self.vision_analyzer = CachedAzureVisionAnalyzer()
        # NT: Initialize object detector for NPC detection
        self.object_detector = ObjectDetector()
        self.reasoner = reasoner
        logger.info("Initialized PerceptionModule with AzureVisionAnalyzer and ObjectDetectorNT")
    
    def _patch_traversability_with_active_tiles(self, traversability_map: List[List[str]], player_map: str, player_x: int, player_y: int) -> List[List[str]]:
        """
        Patch traversability_map with active untraversable tiles from cache.
        
        Only applies tiles from the current map.
        """
        if not traversability_map:
            return traversability_map
        
        if not player_map:
            return traversability_map
        
        # Load active tile index from cache
        NAVIGATION_CACHE_DIR = "./navigation_caches"
        index_path = Path(NAVIGATION_CACHE_DIR) / "active_tile_index.json"
        active_tile_index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    
                # Handle both old list format and new dict format
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
            except Exception as e:
                logger.warning(f"Failed to load active tile index: {e}")
                active_tile_index = {}
        else:
            logger.info("No existing active tile index found")
            active_tile_index = {}
        
        # Use the active_tile_index which has format: {filename: {'class': str, 'tile_pos': [x, y]}}
        if not active_tile_index:
            return traversability_map
        
        try:
            # Iterate over active tiles
            for filename, tile_info in active_tile_index.items():
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
                    if map_name != player_map:
                        continue
                
                tile_pos = tile_info.get('tile_pos', [])
                if len(tile_pos) == 2:
                    tile_x, tile_y = tile_pos
                    
                    # Convert map tile to local grid position (player at 7,7)
                    local_x = 7 + (tile_x - player_x)
                    local_y = 7 + (tile_y - player_y)
                    
                    # Check if tile is in current traversability map bounds
                    if 0 <= local_y < len(traversability_map) and 0 <= local_x < len(traversability_map[0]):
                        # Mark as untraversable
                        traversability_map[local_y][local_x] = '#'
                        logger.debug(f"Patched traversability map: tile ({tile_x}, {tile_y}) marked as untraversable (class={tile_class})")
            
        except Exception as e:
            logger.error(f"Failed to patch traversability map with active tiles: {e}")
        
        return traversability_map
    
    def process(self, game_state: dict) -> PerceptionResult:
        """
        Process game state into structured perception.

        Args:
            game_state: Dict containing 'frame' and other game data

        Returns:
            PerceptionResult with detected objects, OCR text, and navigation targets
            (Note: scene_embedding is a placeholder; actual embeddings handled by memory module)
        """
        frame = np.array(game_state.get('frame'))

        # Check for dialogue or no player visible - bypass Azure vision in these cases
        in_dialogue = detect_dialogue(frame, threshold=0.45)
        player_visible = detect_player_visible(frame)
        
        if USEPERCEPTIONOVERRIDE and (in_dialogue or not player_visible):
            logger.info(f"Bypassing Azure vision analysis: dialogue={in_dialogue}, player_visible={player_visible}")
            detected_objects = []
            ocr_text = []
            navigation_targets = []  # Also bypass navigation target generation
            scene_description = ""
            return PerceptionResult(
                detected_objects=detected_objects,
                ocr_text=ocr_text,
                scene_description=scene_description,
                navigation_targets=navigation_targets,
                llm_outputs={'scene_description': scene_description}
            )
        else:
            # 1. Vision analysis (object detection + OCR in single API call)
            logger.info("Running vision analysis (objects + OCR)")
            vision_result = self.vision_analyzer.analyze(frame)
            detected_objects = vision_result.detected_objects
            ocr_text = vision_result.ocr_text
            logger.info(f"Detected {len(detected_objects)} objects, {len(ocr_text)} OCR text blocks")
            for ocr_text_block in ocr_text:
                if not ocr_text_block.word_level:
                    # NOTE: theres world level OCR and line level OCR, filter properly!
                    logger.debug(f"OCR text: {ocr_text_block.text}")

            # NT: Augment detected objects with NPC detection from object detector
            try:
                npc_detected_objects = self.object_detector.detect_exact_tile_matches(frame)
                if npc_detected_objects:
                    detected_objects.extend(npc_detected_objects)
                    logger.info(f"Added {len(npc_detected_objects)} NPC detections from object detector")
            except Exception as e:
                logger.warning(f"Failed to detect NPC tiles: {e}")

            # 2. Generate scene description (blank for MVP as per requirements)
            scene_description = ""

            # 3. Generate navigation targets
            logger.info("Generating navigation targets")
            navigation_targets = self._generate_navigation_targets(
                detected_objects, game_state
            )
            logger.info(f"Generated {len(navigation_targets)} navigation targets")

            return PerceptionResult(
                detected_objects=detected_objects,
                ocr_text=ocr_text,
                scene_description=scene_description,
                navigation_targets=navigation_targets,
                llm_outputs={'scene_description': scene_description}
            )
    
    def _generate_navigation_targets(
        self, detected_objects: List[DetectedObject], game_state: dict
    ) -> List[NavigationTarget]:
        """
        Generate navigation targets from detected objects and map data.
        
        Uses utility function from existing navigation system.
        
        Args:
            detected_objects: List of detected objects from object detector
            game_state: Game state dict containing map and player data
            
        Returns:
            List of NavigationTarget objects
        """
        # Extract map data and player position from game state
        map_data = game_state.get('map', {})
        traversability_map = get_player_centered_grid(
            map_data=map_data,
            fallback_grid=[['.' for _ in range(15)] for _ in range(15)]
        )

        player_data = game_state.get('player', {})
        position = player_data.get('position', {})
        player_map_tile_pos = (position.get('x', 0), position.get('y', 0))
        player_map_location = player_data.get('location')
        
        # Load portal connections for current map
        portal_connections = load_portal_connections().get(player_map_location, [])
        
        # NT: Update traversability_map from caches
        logger.info("Traversability map before patch:")
        for row in traversability_map:
            logger.info(''.join(row))
        
        traversability_map = self._patch_traversability_with_active_tiles(
            traversability_map, player_map_location, player_map_tile_pos[0], player_map_tile_pos[1]
        )
        
        logger.info("Traversability map after patch:")
        for row in traversability_map:
            logger.info(''.join(row))        # NT: Use NT version of generate_navigation_targets for enhanced target generation
        targets = generate_navigation_targets(
            detected_objects=detected_objects,
            traversability_map=traversability_map,
            player_map_tile_pos=player_map_tile_pos,
            player_map_location=player_map_location,
            portal_connections=portal_connections
        )


        
        return targets or []

