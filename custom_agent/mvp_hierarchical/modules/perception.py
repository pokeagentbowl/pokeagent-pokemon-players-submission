"""Perception module - object detection, scene understanding, and navigation targets.

Heavily modeled after custom_agent/navigation_agent.py with adaptations for hierarchical agent.
"""

import numpy as np
import logging
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from custom_utils.azure_vision_analyzer import DetectedObject, OCRTextBlock
from custom_utils.navigation_targets import NavigationTarget, generate_navigation_targets
from custom_utils.cached_azure_vision_analyzer import CachedAzureVisionAnalyzer
from custom_utils.map_extractor import get_player_centered_grid

logger = logging.getLogger(__name__)


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
        self.reasoner = reasoner
        logger.info("Initialized PerceptionModule with AzureVisionAnalyzer")
    
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
        
        # Generate targets using existing utility
        targets = generate_navigation_targets(
            detected_objects=detected_objects,
            traversability_map=traversability_map,
            player_map_tile_pos=player_map_tile_pos,
            player_map_location=player_map_location
        )
        
        return targets or []

