"""
Label Traversable Areas and Generate Tilemaps

This module provides functions to:
1. Load traversability cluster and entity mappings
2. Match tiles from images to unique tile templates
3. Generate traversability maps with labeled tiles
4. Create visual colormaps with borders
5. Export JSON tilemaps and entity positions

Functions:
- load_cluster_mapping(): Load cluster-to-label-type mapping from JSON
  Input: clustermap_path (str)
  Output: Dict[int, str] cluster_id -> label_type mapping

- load_entity_mapping(): Load cluster-to-entity-type mapping from JSON
  Input: entity_map_path (str)
  Output: Dict[int, int] cluster_id -> entity_type_id mapping

- load_processed_tiles(): Load unique tiles data from JSON
  Input: unique_tiles_dir (str)
  Output: List[Dict] tile data with filename, cluster_id, unique_id

- is_solid_color_tile(): Check if tile is solid color
  Input: tile_array (np.ndarray), color_value (int), tolerance (int)
  Output: bool (True if solid color)

- compute_simple_tile_features(): Compute color histogram + edge features
  Input: img (PIL Image)
  Output: np.ndarray feature vector (53 dimensions)

- find_best_tile_match(): Find best matching tile using features
  Input: tile_img (PIL Image), processed_data (List[Dict]), tile_feature_cache (Dict), match_threshold (float)
  Output: Tuple (cluster_id, unique_id, filename, score)

- build_traversability_map_from_templates(): Build traversability map from templates
  Input: img (PIL Image), base_x (int), base_y (int), processed_data (List[Dict]), cluster_mapping (Optional[Dict]), tile_feature_cache (Optional[Dict])
  Output: Dict cluster_id -> list of tile matches

- generate_traversability_tilemap(): Generate text tilemap from traversability data
  Input: traversability_map (Dict), image_path (Optional[str]), base_x (int), base_y (int), cluster_mapping (Optional[Dict])
  Output: Dict with 'grid', 'symbols', 'min_x', 'min_y', etc.

- create_traversability_colormap(): Create visual colormap with borders
  Input: traversability_map (Dict), frame_pil (PIL Image), cluster_mapping (Optional[Dict]), base_x (int), base_y (int)
  Output: PIL Image with colored borders

- identify_entities(): Identify entities from traversability map
  Input: traversability_map (Dict), entity_mapping (Dict), base_x (int), base_y (int)
  Output: List[Dict] entity data with positions and bounds

- save_traversability_outputs(): Save tilemap, colormap, and entity data
  Input: traversability_map (Dict), tilemap_data (Dict), entities_data (List[Dict]), output_dir (str), uid (str), frame_pil (Optional[PIL Image]), map_name (Optional[str]), cluster_mapping (Optional[Dict]), base_x (int), base_y (int)
  Output: Dict[str, str] paths to saved files

- load_traversability_map(): Load traversability map data
  Input: map_uid (str), traversability_dir (str)
  Output: Optional[Dict] map data

- create_traversability_map_from_image(): Create traversability map from image
  Input: image_path (str), output_dir (str), uid (str), base_x (int), base_y (int), frame_pil (Optional[PIL Image]), map_name (Optional[str]), unique_tiles_dir (str)
  Output: Optional[str] path to created map
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
from collections import defaultdict
from skimage import measure
from typing import Dict, List, Tuple, Optional, Set, Union
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Default paths
# DEFAULT_UNIQUE_TILES_DIR = "pokeAI/object_detection/raw2/"
DEFAULT_UNIQUE_TILES_DIR = "./uniqueembeddings"
DEFAULT_TRAVERSABILITY_DIR = "./traversability_maps"

# Entity type mapping
ENTITY_TYPES = {
    0: 'pokecen',    # Pokemon Center - 4x4 building with door
    1: 'pokemart',   # Pokemart - 4x4 building with door
    2: 'npc',        # Non-player character - single tile
    3: 'house',      # House - 3x3 building with door
    4: 'gym',        # Pokemon Gym - 4x4 building with door
    5: 'exit',       # Exit/Portal - single tile (traversable portal)
    6: 'wall',       # Wall - single tile obstacle (untraversable)
    7: 'grass',      # Tall grass - single tile terrain (traversable)
    8: 'walkable',   # Walkable terrain - single tile
    9: 'water'       # Water - single tile terrain
}

# Entity identification logic
ENTITY_LOGIC = {
    'pokecen': {
        'marker_position': 'bottom_right',
        'bounds': (-3, -3, 0, 0),
        'label_type': 'untraversable',
        'sub_entities': [{'name': 'door', 'offset': (0, -2), 'label_type': 'portal'}]
    },
    'pokemart': {
        'marker_position': 'bottom_right',
        'bounds': (-3, -3, 0, 0),
        'label_type': 'untraversable',
        'sub_entities': [{'name': 'door', 'offset': (0, -2), 'label_type': 'portal'}]
    },
    'gym': {
        'marker_position': 'bottom_right',
        'bounds': (-3, -3, 0, 0),
        'label_type': 'untraversable',
        'sub_entities': [{'name': 'door', 'offset': (0, -2), 'label_type': 'portal'}]
    },
    'house': {
        'marker_position': 'bottom_right',
        'bounds': (-2, -2, 0, 0),
        'label_type': 'untraversable',
        'sub_entities': [{'name': 'door', 'offset': (0, -1), 'label_type': 'portal'}]
    },
    'npc': {
        'marker_position': 'center',
        'bounds': (0, 0, 0, 0),
        'label_type': 'npc',
        'sub_entities': []
    },
    'exit': {
        'marker_position': 'center',
        'bounds': (0, 0, 0, 0),
        'label_type': 'portal',
        'sub_entities': []
    },
    'grass': {
        'marker_position': 'center',
        'bounds': (0, 0, 0, 0),
        'label_type': 'traversable',
        'sub_entities': []
    },
    'tree': {
        'marker_position': 'center',
        'bounds': (0, 0, 0, 0),
        'label_type': 'tree',
        'sub_entities': []
    }
}

# Label type to symbol mapping
LABEL_TYPE_TO_SYMBOL = {
    'traversable': '_',
    'untraversable': '#',
    'water': 'W',
    'npc': 'N',
    'portal': 'A',
    'tree': '#'
}

# Colors for borders
BORDER_COLORS = {
    'traversable': 'gray',
    'untraversable': 'black',
    'water': 'blue',
    'npc': 'cyan',
    'portal': 'green',
    'tree': 'darkgreen',
    'unknown': 'yellow'
}


def load_cluster_mapping(clustermap_path: str) -> Dict[int, str]:
    """
    Load cluster-to-label-type mapping from JSON file.
    
    Args:
        clustermap_path: Path to clustermap.json
        
    Returns:
        Dict mapping cluster_id (int) -> label_type (str)
        
    Expected JSON format:
    {
        "2": "tree",
        "4": "water",
        "8": "npc",
        "12": "untraversable",
        "18": "portal"
    }
    """
    with open(clustermap_path, 'r') as f:
        mapping_str = json.load(f)
    
    cluster_mapping = {int(k): v for k, v in mapping_str.items()}
    #logger.info(f"Loaded cluster mapping: {cluster_mapping}")
    return cluster_mapping


def load_entity_mapping(entity_map_path: str) -> Dict[int, int]:
    """
    Load cluster-to-entity-type mapping from JSON file.
    
    Args:
        entity_map_path: Path to entitymap.json
        
    Returns:
        Dict mapping cluster_id (int) -> entity_type_id (int)
        
    Expected JSON format:
    {
        "88": 0,  # Pokemon Center
        "90": 1,  # Pokemart
        "45": 2   # NPC
    }
    """
    with open(entity_map_path, 'r') as f:
        mapping_str = json.load(f)
    
    entity_mapping = {int(k): int(v) for k, v in mapping_str.items()}
    #logger.info(f"Loaded entity mapping: {entity_mapping}")
    return entity_mapping


def load_processed_tiles(unique_tiles_dir: str = DEFAULT_UNIQUE_TILES_DIR, match_class: Optional[str] = None) -> List[Dict]:
    """
    Load unique tiles data from unique_tiles.json.
    
    Args:
        unique_tiles_dir: Directory containing unique_tiles.json
        match_class: Optional class to filter tiles by (e.g., "npc"). If provided, only tiles
                    belonging to clusters mapped to this class will be returned.
        
    Returns:
        List of tile data dictionaries with 'filename', 'cluster_id', 'unique_id'
    """
    json_path = Path(unique_tiles_dir) / "unique_tiles.json"
    
    if not json_path.exists():
        logger.error(f"unique_tiles.json not found at {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        tile_data = json.load(f)
    
    # Filter by match_class if specified
    if match_class is not None:
        entitymap_path = Path(unique_tiles_dir) / "entitymap.json"
        if entitymap_path.exists():
            entity_mapping = load_entity_mapping(str(entitymap_path))
            # Filter tiles to only include those whose cluster_id maps to an entity that matches match_class
            filtered_tiles = []
            for tile in tile_data:
                cluster_id = tile.get('cluster_id')
                if cluster_id is not None:
                    entity_type_id = entity_mapping.get(cluster_id)
                    if entity_type_id is not None:
                        entity_name = ENTITY_TYPES.get(entity_type_id, 'unknown')
                        if entity_name == match_class:
                            filtered_tiles.append(tile)
            tile_data = filtered_tiles
            #logger.info(f"Filtered tiles by class '{match_class}': {len(tile_data)} tiles remaining")
        else:
            logger.warning(f"entitymap.json not found at {entitymap_path}, cannot filter by match_class")
    
    # Add source directory for each tile
    for tile in tile_data:
        tile['source_dir'] = str(Path(unique_tiles_dir))
    
    #logger.info(f"Loaded {len(tile_data)} unique tiles from {json_path}")
    return tile_data


def is_solid_color_tile(tile_array: np.ndarray, color_value: int = 255, tolerance: int = 5) -> bool:
    """
    Check if a tile is solid color (white or black).
    
    Args:
        tile_array: numpy array of the tile
        color_value: 255 for white, 0 for black
        tolerance: allowed deviation
        
    Returns:
        True if tile is solid color within tolerance
    """
    if len(tile_array.shape) == 3:
        tile_array = cv2.cvtColor(tile_array, cv2.COLOR_RGB2GRAY)
    return np.all(np.abs(tile_array - color_value) <= tolerance)


def compute_simple_tile_features(img: Image.Image) -> np.ndarray:
    """
    Compute combined features for tile matching: color histogram + edge features.
    
    Args:
        img: PIL Image of 32x32 tile
        
    Returns:
        Feature vector (48 color histogram + 5 edge features = 53 dims)
    """
    img_rgb = img.convert('RGB')
    img_array = np.array(img_rgb)
    
    # Color histogram (3 channels x 16 bins = 48 features)
    hist_r = cv2.calcHist([img_array], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_array], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_array], [2], None, [16], [0, 256]).flatten()
    color_hist = np.concatenate([hist_r, hist_g, hist_b])
    
    # Edge features (5 features)
    img_gray = img.convert('L')
    img_array_gray = np.array(img_gray)
    
    edges = cv2.Canny(img_array_gray, 100, 200)
    edge_density = np.sum(edges > 0) / (32 * 32)
    
    sobelx = cv2.Sobel(img_array_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_array_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_x_mean = np.mean(np.abs(sobelx))
    grad_y_mean = np.mean(np.abs(sobely))
    grad_x_std = np.std(sobelx)
    grad_y_std = np.std(sobely)
    
    edge_feat = np.array([edge_density, grad_x_mean, grad_y_mean, grad_x_std, grad_y_std])
    
    return np.concatenate([color_hist, edge_feat])


def find_best_tile_match(tile_img: Image.Image, processed_data: List[Dict], 
                         tile_feature_cache: Dict, match_threshold: float = 0.85,
                         return_features: bool = False) -> Tuple:
    """
    Find the best matching tile using simple feature similarity.
    
    Args:
        tile_img: PIL Image of the tile to match (32x32)
        processed_data: List of tile data dictionaries
        tile_feature_cache: Dict caching computed features by filename
        match_threshold: Minimum similarity score (0-1)
        return_features: If True, return tile features as 5th element
        
    Returns:
        (cluster_id, unique_id, filename, best_score, tile_features) if return_features=True,
        else (cluster_id, unique_id, filename, best_score)
        cluster_id is None if no match found above threshold
    """
    tile_features = compute_simple_tile_features(tile_img)
    
    best_score = -1.0
    best_cluster_id = None
    best_unique_id = None
    best_filename = None
    
    for item in processed_data:
        filename = item['filename']
        
        # Compute or retrieve cached features
        if filename not in tile_feature_cache:
            source_dir = Path(item.get('source_dir', DEFAULT_UNIQUE_TILES_DIR))
            template_path = source_dir / filename
            
            if not template_path.exists():
                continue
            
            template_img = Image.open(template_path)
            template_features = compute_simple_tile_features(template_img)
            tile_feature_cache[filename] = template_features
        
        template_features = tile_feature_cache[filename]
        
        # Compute cosine similarity
        similarity = np.dot(tile_features, template_features) / (
            np.linalg.norm(tile_features) * np.linalg.norm(template_features) + 1e-10
        )
        
        if similarity > best_score:
            best_score = similarity
            if similarity >= match_threshold:
                best_cluster_id = item['cluster_id']
                best_unique_id = item.get('unique_id', 0)
                best_filename = filename
    
    if return_features:
        return best_cluster_id, best_unique_id, best_filename, best_score, tile_features
    else:
        return best_cluster_id, best_unique_id, best_filename, best_score


def build_traversability_map_from_templates(img: Image.Image, base_x: int, base_y: int, 
                                            processed_data: List[Dict],
                                            cluster_mapping: Optional[Dict] = None,
                                            tile_feature_cache: Optional[Dict] = None,
                                            return_features: bool = False) -> Dict:
    """
    Build a traversability map by chopping image into 32x32 tiles and finding matches.
    
    Args:
        img: PIL Image to process
        base_x: Base tile x coordinate
        base_y: Base tile y coordinate
        processed_data: List of tile data dictionaries
        cluster_mapping: Optional dict mapping cluster_id -> label_type
        tile_feature_cache: Optional cache for tile features
        return_features: If True, include 'features' key in match dictionaries
        
    Returns:
        Dict mapping cluster_id -> list of match dictionaries with:
            - 'local_x': x position in pixels
            - 'local_y': y position in pixels
            - 'score': match score
            - 'match_type': 'threshold' or 'unknown'
            - 'unique_id': tile unique ID
            - 'filename': tile filename
            - 'features': feature vector (only if return_features=True)
    """
    if tile_feature_cache is None:
        tile_feature_cache = {}
    
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    
    traversability_map = defaultdict(list)
    
    #logger.info(f"Chopping image ({img_width}x{img_height}) into 32x32 tiles...")
    
    total_tiles = 0
    skipped_white = 0
    skipped_black = 0
    matched_tiles = 0
    unknown_tiles = 0
    
    # Chop image into 32x32 tiles
    for y in range(0, img_height, 32):
        for x in range(0, img_width, 32):
            if x + 32 > img_width or y + 32 > img_height:
                continue
            
            tile_array = img_array[y:y+32, x:x+32]
            total_tiles += 1
            
            # Skip solid white tiles
            if is_solid_color_tile(tile_array, 255, tolerance=5):
                skipped_white += 1
                continue
            
            # Skip solid black tiles
            if is_solid_color_tile(tile_array, 0, tolerance=5):
                skipped_black += 1
                continue
            
            # Extract tile as PIL Image
            tile_img = img.crop((x, y, x+32, y+32))
            
            # Find best match
            if return_features:
                cluster_id, unique_id, filename, best_score, tile_features = find_best_tile_match(
                    tile_img, processed_data, tile_feature_cache, match_threshold=0.85, return_features=True
                )
            else:
                cluster_id, unique_id, filename, best_score = find_best_tile_match(
                    tile_img, processed_data, tile_feature_cache, match_threshold=0.85
                )
                tile_features = None
            
            if cluster_id is not None:
                # Skip if cluster mapping provided and this cluster not in it
                if cluster_mapping is not None and cluster_id not in cluster_mapping:
                    continue
                
                matched_tiles += 1
                match_entry = {
                    'local_x': int(x),
                    'local_y': int(y),
                    'score': float(best_score),
                    'match_type': 'threshold',
                    'unique_id': unique_id,
                    'filename': filename
                }
                if return_features and tile_features is not None:
                    match_entry['features'] = tile_features.tolist()
                traversability_map[cluster_id].append(match_entry)
            else:
                # Unknown tile
                unknown_tiles += 1
                match_entry = {
                    'local_x': int(x),
                    'local_y': int(y),
                    'score': float(best_score),
                    'match_type': 'unknown',
                    'unique_id': -1
                }
                if return_features and tile_features is not None:
                    match_entry['features'] = tile_features.tolist()
                traversability_map[-999].append(match_entry)
    
    #logger.info(f"Tile matching complete: {total_tiles} total, {skipped_white} white, "
                #f"{skipped_black} black, {matched_tiles} matched, {unknown_tiles} unknown")
    
    return traversability_map


def generate_traversability_tilemap(traversability_map: Dict, image_path: Optional[str] = None,
                                    base_x: int = 0, base_y: int = 0,
                                    cluster_mapping: Optional[Dict] = None) -> Dict:
    """
    Generate text tilemap from traversability map data.
    
    Args:
        traversability_map: Dict mapping cluster_id -> list of matches
        image_path: Optional path to image (for metadata)
        base_x: Base tile x coordinate
        base_y: Base tile y coordinate
        cluster_mapping: Dict mapping cluster_id -> label_type
        
    Returns:
        Dict with 'tilemap' (2D array of symbols) and metadata
    """
    if not traversability_map:
        return {'tilemap': [], 'base_x': base_x, 'base_y': base_y}
    
    # Determine grid dimensions
    max_x = max_y = 0
    for matches in traversability_map.values():
        for match in matches:
            tile_x = match['local_x'] // 32
            tile_y = match['local_y'] // 32
            max_x = max(max_x, tile_x)
            max_y = max(max_y, tile_y)
    
    grid_width = max_x + 1
    grid_height = max_y + 1
    
    # Initialize grid with traversable
    tilemap = [['_' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Fill in symbols
    for cluster_id, matches in traversability_map.items():
        if cluster_id == -999:  # Unknown
            symbol = '?'
        else:
            label_type = cluster_mapping.get(cluster_id, 'unknown') if cluster_mapping else 'unknown'
            symbol = LABEL_TYPE_TO_SYMBOL.get(label_type, '?')
        
        for match in matches:
            tile_x = match['local_x'] // 32
            tile_y = match['local_y'] // 32
            tilemap[tile_y][tile_x] = symbol
    
    return {
        'tilemap': tilemap,
        'base_x': base_x,
        'base_y': base_y,
        'width': grid_width,
        'height': grid_height,
        'image': image_path
    }


def create_traversability_colormap(traversability_map: Dict, frame_pil: Image.Image,
                                   cluster_mapping: Optional[Dict] = None,
                                   base_x: int = 0, base_y: int = 0) -> Image.Image:
    """
    Create a visual colormap with borders and crosses for traversability.
    
    Args:
        traversability_map: Dict mapping cluster_id -> list of matches
        frame_pil: Original frame PIL Image
        cluster_mapping: Dict mapping cluster_id -> label_type
        base_x: Base tile x coordinate for global coordinate labeling
        base_y: Base tile y coordinate for global coordinate labeling
        
    Returns:
        PIL Image with visual borders and markings
    """
    from skimage import measure
    
    labeled_img = frame_pil.copy()
    draw = ImageDraw.Draw(labeled_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    sub_tile_pixels = 32
    
    # Get dimensions
    img_array = np.array(frame_pil)
    grid_height = img_array.shape[0] // sub_tile_pixels
    grid_width = img_array.shape[1] // sub_tile_pixels
    
    # Create masks for each label type
    masks = {
        'untraversable': np.zeros((grid_height, grid_width), dtype=bool),
        'water': np.zeros((grid_height, grid_width), dtype=bool),
        'npc': np.zeros((grid_height, grid_width), dtype=bool),
        'portal': np.zeros((grid_height, grid_width), dtype=bool),
        'tree': np.zeros((grid_height, grid_width), dtype=bool),
        'traversable': np.zeros((grid_height, grid_width), dtype=bool),
        'unknown': np.zeros((grid_height, grid_width), dtype=bool)
    }
    
    # Populate masks
    for cluster_id, matches in traversability_map.items():
        if cluster_id == -999:
            label_type = 'unknown'
        else:
            label_type = cluster_mapping.get(cluster_id, 'unknown') if cluster_mapping else 'unknown'
        
        for match in matches:
            tile_x = match['local_x'] // sub_tile_pixels
            tile_y = match['local_y'] // sub_tile_pixels
            
            if 0 <= tile_y < grid_height and 0 <= tile_x < grid_width:
                if label_type in masks:
                    masks[label_type][tile_y, tile_x] = True
    
    # For each type, find connected components and draw borders
    for label_type, mask in masks.items():
        if not mask.any():
            continue
        
        # Find connected components
        labeled_mask = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            pixel_min_x = min_col * sub_tile_pixels
            pixel_max_x = (max_col) * sub_tile_pixels
            pixel_min_y = min_row * sub_tile_pixels
            pixel_max_y = (max_row) * sub_tile_pixels
            
            # Draw border
            color = BORDER_COLORS.get(label_type, 'gray')
            draw.rectangle([pixel_min_x, pixel_min_y, pixel_max_x, pixel_max_y], outline=color, width=2)
            
            # For untraversable, draw crosses in each tile of the region
            if label_type == 'untraversable':
                for r in range(min_row, max_row):
                    for c in range(min_col, max_col):
                        if r < grid_height and c < grid_width and labeled_mask[r, c] == region.label:
                            px = c * sub_tile_pixels
                            py = r * sub_tile_pixels
                            # Draw cross
                            draw.line([px, py, px + sub_tile_pixels, py + sub_tile_pixels], fill='black', width=1)
                            draw.line([px + sub_tile_pixels, py, px, py + sub_tile_pixels], fill='black', width=1)
            
            # For trees, draw green crosses in each tile of the region
            elif label_type == 'tree':
                for r in range(min_row, max_row):
                    for c in range(min_col, max_col):
                        if r < grid_height and c < grid_width and labeled_mask[r, c] == region.label:
                            px = c * sub_tile_pixels
                            py = r * sub_tile_pixels
                            # Draw green cross
                            draw.line([px, py, px + sub_tile_pixels, py + sub_tile_pixels], fill='green', width=2)
                            draw.line([px + sub_tile_pixels, py, px, py + sub_tile_pixels], fill='green', width=2)
            
            # For NPCs, draw a single filled circle in the center of the region
            elif label_type == 'npc':
                # Calculate center of the entire region
                region_center_x = (min_col + max_col) * sub_tile_pixels // 2
                region_center_y = (min_row + max_row) * sub_tile_pixels // 2
                radius = 12  # Larger radius for region center
                draw.ellipse([region_center_x - radius, region_center_y - radius, 
                            region_center_x + radius, region_center_y + radius], 
                           fill='cyan', outline='blue', width=2)
            
            # For portals, draw a thick green border around the region and diagonal lines from center
            elif label_type == 'portal':
                # Draw thick border around entire region
                draw.rectangle([pixel_min_x, pixel_min_y, pixel_max_x, pixel_max_y], 
                             outline='green', width=4)
                # Draw diagonal arrows from center of region
                region_center_x = (min_col + max_col) * sub_tile_pixels // 2
                region_center_y = (min_row + max_row) * sub_tile_pixels // 2
                arrow_size = 12
                draw.line([region_center_x - arrow_size, region_center_y - arrow_size, 
                          region_center_x + arrow_size, region_center_y + arrow_size], 
                         fill='green', width=3)
                draw.line([region_center_x - arrow_size, region_center_y + arrow_size, 
                          region_center_x + arrow_size, region_center_y - arrow_size], 
                         fill='green', width=3)
            
            # For unknown tiles, draw a yellow border and big "?" mark in center of region
            elif label_type == 'unknown':
                # Draw thick yellow border around entire region
                draw.rectangle([pixel_min_x, pixel_min_y, pixel_max_x, pixel_max_y], 
                             outline='yellow', width=4)
                # Draw big "?" in center of region
                region_center_x = (min_col + max_col) * sub_tile_pixels // 2
                region_center_y = (min_row + max_row) * sub_tile_pixels // 2
                
                # Try to use a large font for the "?"
                try:
                    big_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    big_font = font
                
                text = "?"
                # Get text size to center it
                try:
                    bbox = draw.textbbox((0, 0), text, font=big_font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_x = region_center_x - text_width // 2
                    text_y = region_center_y - text_height // 2
                    # Draw white background for visibility
                    draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], 
                                 fill='white')
                    draw.text((text_x, text_y), text, fill='orange', font=big_font)
                except:
                    # Fallback
                    draw.text((region_center_x - 6, region_center_y - 10), text, fill='orange', font=big_font)
    
    # Add coordinate labels using global tile coordinates
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        small_font = ImageFont.load_default()
    
    # Draw Y-axis labels on the left (global tile Y coordinates)
    for sub_y in range(grid_height):
        global_y = base_y + sub_y
        py = sub_y * sub_tile_pixels + sub_tile_pixels // 2
        text = str(global_y)
        
        # Draw with white background for visibility
        try:
            bbox = draw.textbbox((2, py - 6), text, font=small_font)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill='white', outline='black')
            draw.text((2, py - 6), text, fill='red', font=small_font)
        except:
            draw.text((2, py - 6), text, fill='red', font=small_font)
    
    # Draw X-axis labels on the top (global tile X coordinates)
    for sub_x in range(grid_width):
        global_x = base_x + sub_x
        px = sub_x * sub_tile_pixels + sub_tile_pixels // 2
        text = str(global_x)
        
        # Draw with white background for visibility
        try:
            bbox = draw.textbbox((px - 6, 2), text, font=small_font)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill='white', outline='black')
            draw.text((px - 6, 2), text, fill='blue', font=small_font)
        except:
            draw.text((px - 6, 2), text, fill='blue', font=small_font)
    
    return labeled_img


def identify_entities(traversability_map: Dict, entity_mapping: Dict,
                     base_x: int = 0, base_y: int = 0) -> List[Dict]:
    """
    Identify entities based on cluster matches and entity mapping.
    
    Args:
        traversability_map: Dict mapping cluster_id -> list of matches
        entity_mapping: Dict mapping cluster_id -> entity_type_id
        base_x: Base tile x coordinate (global)
        base_y: Base tile y coordinate (global)
        
    Returns:
        List of entity dictionaries with positions, bounds, and sub-entities
    """
    entities = []
    tile_size = 32
    
    #logger.info(f"Identifying entities from {len(entity_mapping)} entity clusters...")
    
    for cluster_id, entity_type_id in entity_mapping.items():
        if cluster_id not in traversability_map:
            continue
        
        entity_name = ENTITY_TYPES.get(entity_type_id, f'unknown_{entity_type_id}')
        entity_logic = ENTITY_LOGIC.get(entity_name, {
            'marker_position': 'center',
            'bounds': (0, 0, 0, 0),
            'sub_entities': []
        })
        
        matches = traversability_map[cluster_id]
        
        for match_info in matches:
            marker_x = match_info['local_x'] // tile_size
            marker_y = match_info['local_y'] // tile_size
            
            global_marker_x = base_x + marker_x
            global_marker_y = base_y + marker_y
            
            bounds = entity_logic['bounds']
            
            if entity_logic['marker_position'] == 'bottom_right':
                entity_x = global_marker_x + bounds[0]
                entity_y = global_marker_y + bounds[1]
                entity_width = abs(bounds[0]) + 1
                entity_height = abs(bounds[1]) + 1
            else:
                entity_x = global_marker_x
                entity_y = global_marker_y
                entity_width = 1
                entity_height = 1
            
            entity_entry = {
                'type': entity_name,
                'type_id': entity_type_id,
                'x': entity_x,
                'y': entity_y,
                'width': entity_width,
                'height': entity_height,
                'marker': {'x': global_marker_x, 'y': global_marker_y},
                'sub_entities': []
            }
            
            # Add sub-entities (e.g., doors)
            for sub in entity_logic['sub_entities']:
                sub_x = global_marker_x + sub['offset'][0]
                sub_y = global_marker_y + sub['offset'][1]
                entity_entry['sub_entities'].append({
                    'name': f"{entity_name}_{sub['name']}",
                    'x': sub_x,
                    'y': sub_y,
                    'label_type': sub['label_type']
                })
            
            entities.append(entity_entry)
    
    #logger.info(f"Identified {len(entities)} entities")
    return entities


def save_traversability_outputs(traversability_map: Dict, tilemap_data: Dict,
                                entities_data: List[Dict], output_dir: str,
                                uid: str, frame_pil: Optional[Image.Image] = None,
                                map_name: Optional[str] = None,
                                cluster_mapping: Optional[Dict] = None,
                                base_x: int = 0, base_y: int = 0) -> Dict[str, str]:
    """
    Save all traversability outputs to files.
    
    Args:
        traversability_map: Dict mapping cluster_id -> list of matches
        tilemap_data: Tilemap dictionary from generate_traversability_tilemap
        entities_data: List of entities from identify_entities
        output_dir: Output directory path
        uid: Unique ID for this map
        frame_pil: Optional frame image to save
        map_name: Optional map name
        cluster_mapping: Optional cluster mapping for colormap
        base_x: Base tile x coordinate for global coordinate labeling
        base_y: Base tile y coordinate for global coordinate labeling
        
    Returns:
        Dict with paths to created files:
            - 'raw_image': path to saved frame
            - 'colormap': path to colormap image
            - 'tilemap': path to tilemap JSON
            - 'entities': path to entities JSON
    """
    output_path = Path(output_dir) / uid
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Save raw image
    if frame_pil is not None:
        raw_path = output_path / f"{uid}_raw.png"
        frame_pil.save(raw_path)
        output_files['raw_image'] = str(raw_path)
        #logger.info(f"Saved raw image: {raw_path}")
    
    # Save colormap
    if frame_pil is not None:
        colormap = create_traversability_colormap(traversability_map, frame_pil, cluster_mapping, base_x, base_y)
        colormap_path = output_path / f"traversability_colourmap_{uid.split('_')[-1]}.png"
        colormap.save(colormap_path)
        output_files['colormap'] = str(colormap_path)
        #logger.info(f"Saved colormap: {colormap_path}")
    
    # Save tilemap JSON
    tilemap_path = output_path / f"traversability_tilemap_{uid.split('_')[-1]}.json"
    with open(tilemap_path, 'w') as f:
        json.dump(tilemap_data, f, indent=2)
    output_files['tilemap'] = str(tilemap_path)
    #logger.info(f"Saved tilemap: {tilemap_path}")
    
    # Save entities JSON
    if entities_data:
        entities_path = output_path / f"{uid.split('_')[-1]}_entities.json"
        entities_output = {
            'map_uid': uid,
            'map_name': map_name,
            'tile_size': 32,
            'entities': entities_data
        }
        with open(entities_path, 'w') as f:
            json.dump(entities_output, f, indent=2)
        output_files['entities'] = str(entities_path)
        #logger.info(f"Saved entities: {entities_path}")
    
    return output_files


def load_traversability_map(map_uid: str, 
                           traversability_dir: str = DEFAULT_TRAVERSABILITY_DIR) -> Optional[Dict]:
    """
    Load a traversability map from disk.
    
    Args:
        map_uid: Unique ID of the map
        traversability_dir: Directory containing traversability maps
        
    Returns:
        Dict with map data including tilemap, entities, image, etc.
    """
    map_dir = Path(traversability_dir) / map_uid
    
    if not map_dir.exists():
        logger.warning(f"Map directory not found: {map_dir}")
        return None
    
    try:
        # Load tilemap JSON - try multiple patterns
        tilemap_files = list(map_dir.glob("*tilemap*.json"))
        if not tilemap_files:
            logger.warning(f"No tilemap JSON found in {map_dir}")
            return None
        
        tilemap_path = tilemap_files[0]
        with open(tilemap_path, 'r') as f:
            tilemap_data = json.load(f)
        
        # Load entities if available
        entities_files = list(map_dir.glob("*_entities.json"))
        entities_data = []
        if entities_files:
            with open(entities_files[0], 'r') as f:
                entities_json = json.load(f)
                entities_data = entities_json.get('entities', [])
        
        # Load raw image
        raw_image_path = map_dir / f"{map_uid}_raw.png"
        map_image = None
        if raw_image_path.exists():
            map_image = Image.open(raw_image_path)
        
        # Load colormap if available
        colormap_files = list(map_dir.glob(f"traversability_colourmap_*.png"))
        colormap_image = None
        if colormap_files:
            colormap_image = Image.open(colormap_files[0])
        
        if 'tilemap' not in tilemap_data:
            tilemap_data['tilemap'] = tilemap_data.get('grid', [])

        map_data = {
            'uid': map_uid,
            'tilemap': tilemap_data.get('tilemap', []),
            'base_x': tilemap_data.get('base_x', 0),
            'base_y': tilemap_data.get('base_y', 0),
            'width': tilemap_data.get('width', 0),
            'height': tilemap_data.get('height', 0),
            'entities': entities_data,
            'image': map_image,
            'colormap': colormap_image,
            'min_x': tilemap_data.get('base_x', 0),
            'min_y': tilemap_data.get('base_y', 0)
        }
        
        #logger.info(f"Loaded traversability map {map_uid}")
        return map_data
        
    except Exception as e:
        logger.error(f"Error loading traversability map {map_uid}: {e}")
        return None

def detect_tile_matches_from_image(frame: Union[np.ndarray, Image.Image], 
                                   tile_database_dir: str = DEFAULT_UNIQUE_TILES_DIR,
                                   match_threshold: float = 0.95, match_class: str = "npc", is_simplefeat: bool = True, is_debug: bool = True) -> List[Dict]:
    """
    Detect exact tile matches in an image and return match data.
    
    Checks both the static tile database and active cache tiles.
    
    Args:
        frame: Input image as numpy array or PIL Image
        tile_database_dir: Directory containing unique tiles data
        match_threshold: Minimum similarity score for matches
        match_class: Class to match ("npc", "untraversable", etc.)
        is_simplefeat: Use simple features (default True)
        is_debug: Save debug visualization (default True)
        
    Returns:
        List of match dictionaries with keys: cluster_id, local_x, local_y, score, etc.
    """
    # Convert to PIL Image if needed
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = frame
    
    # Preprocess image: upscale x2 if frame is too small
    if img.size[1] == 160:
        img = img.resize((480, 320), Image.NEAREST)
    if img.size[1] == 320:
        # Pad the frame with 16px top and bottom only
        padded_height = img.height + 32
        padded_frame = Image.new('RGB', (img.width, padded_height), (0, 0, 0))
        padded_frame.paste(img, (0, 16))
        img = padded_frame
    
    # Load processed tiles from the database directory
    processed_data = load_processed_tiles(tile_database_dir, match_class=match_class)
    
    # Load active cache tiles if available
    active_cache_dir = Path("./navigation_caches")
    active_features_path = active_cache_dir / "active_tile_simplefeatures.json"
    
    # Load active_tile_index.json once for reuse
    active_tile_index = {}
    active_tile_filenames = set()
    index_path = active_cache_dir / "active_tile_index.json"
    if index_path.exists():
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
                # Handle both old list format and new dict format
                if isinstance(index_data, dict):
                    active_tile_index = index_data
                    active_tile_filenames = set(index_data.keys())
                elif isinstance(index_data, list):
                    # Convert old list format to dict
                    for item in index_data:
                        if item.get('filename'):
                            active_tile_index[item['filename']] = item
                            active_tile_filenames.add(item['filename'])
            #logger.info(f"Loaded active tile index with {len(active_tile_index)} entries")
        except Exception as e:
            logger.warning(f"Failed to load active tile index: {e}")
    
    if active_features_path.exists():
        # Load existing cache
        try:
            with open(active_features_path, 'r') as f:
                features_dict = json.load(f)
            
            # features_dict is {filename: features_array}, combine with active_tile_index
            active_features = []
            for filename, features in features_dict.items():
                if filename in active_tile_index:
                    index_entry = active_tile_index[filename]
                    tile_class = index_entry.get('class')
                    tile_pos = index_entry.get('tile_pos', [])
                    
                    if tile_class:  # Only include if we have class info
                        active_features.append({
                            'filename': filename,
                            'features': features,
                            'class': tile_class,
                            'tile_pos': tile_pos
                        })
            
            #logger.info(f"Loaded existing active cache with {len(active_features)} tiles (combined features + index)")
        except Exception as e:
            logger.warning(f"Failed to load active cache: {e}")
            active_features = []
    else:
        # Cache doesn't exist, traverse directory for PNG files and build cache
        #logger.info("Active cache not found, building from PNG files...")
        active_features = []
        
        if active_cache_dir.exists():
            # Find all PNG files in the cache directory
            png_files = list(active_cache_dir.glob("*.png"))
            #logger.info(f"Found {len(png_files)} PNG files in cache directory")
            
            for png_path in png_files:
                try:
                    # Load tile image
                    tile_img = Image.open(png_path)
                    
                    # Compute features
                    features = compute_simple_tile_features(tile_img)
                    
                    # Try to load class and position from active_tile_index.json
                    filename = png_path.name
                    tile_class = None
                    tile_pos = []
                    
                    # Use pre-loaded active_tile_index
                    if filename in active_tile_index:
                        item = active_tile_index[filename]
                        # Support both old format (is_npc) and new format (class)
                        if 'class' in item:
                            tile_class = item['class']
                        elif 'is_npc' in item:
                            tile_class = 'npc' if item['is_npc'] else 'untraversable'
                        
                        # Try to extract position from filename if not in index
                        if 'tile_pos' in item and item['tile_pos']:
                            tile_pos = item['tile_pos']
                        else:
                            # Extract from filename: {mapx}_{mapy}.png
                            try:
                                parts = filename.replace('.png', '').split('_')
                                if len(parts) >= 2:
                                    tile_pos = [int(parts[-2]), int(parts[-1])]
                            except:
                                tile_pos = []
                    
                    # If we couldn't determine class from index, skip this tile
                    if tile_class is None:
                        logger.warning(f"Could not determine class for {filename}, skipping")
                        continue
                    
                    # Add to active features
                    active_features.append({
                        'filename': filename,
                        'features': features.tolist(),
                        'class': tile_class,
                        'tile_pos': tile_pos
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process {png_path}: {e}")
                    continue
            
            # Save the newly built cache
            try:
                # Convert to filename -> features format for saving
                features_dict = {item['filename']: item['features'] for item in active_features}
                with open(active_features_path, 'w') as f:
                    json.dump(features_dict, f, indent=2)
                #logger.info(f"Saved new active cache with {len(active_features)} tiles")
            except Exception as e:
                logger.error(f"Failed to save active cache: {e}")
        else:
            logger.warning(f"Active cache directory {active_cache_dir} does not exist")
            active_features = []
    
    # Load tile feature cache early (needed for override cache computation)
    tile_feature_cache = {}
    tile_feature_cache_path = Path(tile_database_dir) / "tile_simplefeature_cache.json" if is_simplefeat else Path(tile_database_dir) / "tile_feature_cache.json"
    if tile_feature_cache_path.exists():
        try:
            with open(tile_feature_cache_path, 'r') as f:
                tile_feature_cache = json.load(f)
            #logger.info(f"Loaded tile feature cache with {len(tile_feature_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load tile feature cache: {e}")
    
    # Load or create override_unique_tiles.json
    # This tracks when active tiles exactly match (>0.99) unique tiles,
    # allowing active tiles to override the unique tile's class
    override_cache_path = active_cache_dir / "override_unique_tiles.json"
    override_cache = {}
    
    if override_cache_path.exists():
        try:
            with open(override_cache_path, 'r') as f:
                override_cache = json.load(f)
            #logger.info(f"Loaded override cache with {len(override_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load override cache: {e}")
            override_cache = {}
    
    # Update override cache by checking exact matches between active tiles and unique tiles
    # Only check for grass, npc, and untraversable classes
    override_classes = ['grass', 'npc', 'untraversable']
    override_threshold = 0.99
    
    for active_tile in active_features:
        active_class = active_tile.get('class')
        if active_class not in override_classes:
            continue
        
        active_filename = active_tile['filename']
        active_features_array = np.array(active_tile['features'])
        
        # Check against all unique tiles in processed_data
        for unique_tile in processed_data:
            if unique_tile.get('is_active'):
                continue  # Skip other active tiles
            
            unique_filename = unique_tile['filename']
            unique_cluster_id = unique_tile['cluster_id']
            
            # Get or compute unique tile features
            if 'features' in unique_tile:
                unique_features_array = np.array(unique_tile['features'])
            else:
                # Load from cache or compute
                if unique_filename in tile_feature_cache:
                    unique_features_array = np.array(tile_feature_cache[unique_filename])
                else:
                    # Load the unique tile image and compute features
                    try:
                        unique_tile_path = Path(tile_database_dir) / unique_filename
                        if unique_tile_path.exists():
                            unique_tile_img = Image.open(unique_tile_path)
                            unique_features_array = compute_simple_tile_features(unique_tile_img)
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Could not load unique tile {unique_filename}: {e}")
                        continue
            
            # Compute cosine similarity
            similarity = np.dot(active_features_array, unique_features_array) / (
                np.linalg.norm(active_features_array) * np.linalg.norm(unique_features_array) + 1e-10
            )
            
            # If exact match (>0.99), add to override cache
            if similarity >= override_threshold:
                # Map entity class name to entity_type_id
                entity_type_id = None
                for et_id, et_name in ENTITY_TYPES.items():
                    if et_name == active_class:
                        entity_type_id = et_id
                        break
                
                override_key = unique_filename
                override_cache[override_key] = {
                    'active_tile_name': active_filename,
                    'unique_tile_name': unique_filename,
                    'unique_tile_cluster_id': int(unique_cluster_id),
                    'new_class': active_class,
                    'new_entity_type_id': entity_type_id,
                    'similarity': float(similarity)
                }
                
                #logger.info(f"Override: {unique_filename} (cluster {unique_cluster_id}) -> {active_filename} (class: {active_class}, similarity: {similarity:.4f})")
    
    # Save updated override cache
    if override_cache:
        try:
            override_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(override_cache_path, 'w') as f:
                json.dump(override_cache, f, indent=2)
            #logger.info(f"Saved override cache with {len(override_cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save override cache: {e}")
    
    # Apply override cache: remove unique tiles that are overridden by active tiles
    original_count = len(processed_data)
    processed_data = [
        tile for tile in processed_data 
        if tile['filename'] not in override_cache or tile.get('is_active', False)
    ]
    removed_count = original_count - len(processed_data)
    #if removed_count > 0:
        #logger.info(f"Removed {removed_count} unique tiles that are overridden by active cache tiles")
    
    # Add active tiles to processed_data (only those matching the requested class)
    for tile_data in active_features:
        if tile_data.get('class') == match_class:
            # Convert to processed_data format
            processed_data.append({
                'filename': tile_data['filename'],
                'cluster_id': -1,  # Placeholder for active tiles
                'unique_id': tile_data['filename'],
                'features': tile_data['features'],
                'is_active': True  # Mark as active cache tile
            })
    
    #logger.info(f"Loaded {len([t for t in active_features if t.get('class') == match_class])} active cache tiles for class '{match_class}'")
    
    # Load entity mapping for validation
    entitymap_path = Path(tile_database_dir) / "entitymap.json"
    entity_mapping = {}
    if entitymap_path.exists():
        entity_mapping = load_entity_mapping(str(entitymap_path))

    # Build traversability map using template matching
    traversability_map = build_traversability_map_from_templates(
        img, base_x=0, base_y=0, processed_data=processed_data, 
        cluster_mapping=None, tile_feature_cache=tile_feature_cache,
        return_features=True  # Include features for neighbor similarity calculations
    )
    
    # Collect all matches above threshold
    detected_matches = []

    # Filter by entity mapping for valid cluster_ids in match_class
    valid_cluster_ids = {-1}  # Include -1 for active cache tiles by default
    if entity_mapping:
        # Get the entity_type_id for match_class
        entity_type_id = None
        for et_id, et_name in ENTITY_TYPES.items():
            if et_name == match_class:
                entity_type_id = et_id
                break
        
        if entity_type_id is not None:
            for cluster_id, cluster_entity_type_id in entity_mapping.items():
                if cluster_entity_type_id == entity_type_id:
                    valid_cluster_ids.add(cluster_id)
    
    for cluster_id, matches in traversability_map.items():
        if cluster_id not in valid_cluster_ids:
            continue
        for match in matches:
            if match['score'] >= match_threshold:
                match_data = match.copy()
                match_data['cluster_id'] = cluster_id
                detected_matches.append(match_data)
    if not detected_matches:
        print(f"DEBUG AVG MATCH SCORE: {np.mean([m['score'] for matches in traversability_map.values() for m in matches]):.4f}")
    
    # Filter out isolated tiles (those without enough neighbors)
    # A tile is kept if it has at least 2 neighbors with similarity > 0.8 within +/- 2 tiles in x and y
    # Recalculate similarity between reference tile and each neighbor using simple features
    filtered_out_set = set()
    tile_size = 32
    neighbor_threshold = 0.8
    min_neighbors = 2
    
    for i, tile in enumerate(detected_matches):
        tile_x = tile['local_x'] // tile_size
        tile_y = tile['local_y'] // tile_size
        
        # Get or compute reference tile features
        if 'features' in tile:
            ref_features = np.array(tile['features'])
        else:
            ref_tile_img = img.crop((tile['local_x'], tile['local_y'], 
                                     tile['local_x'] + tile_size, tile['local_y'] + tile_size))
            ref_features = compute_simple_tile_features(ref_tile_img)
        
        # Count valid neighbors within +/- 2 tiles
        valid_neighbors = 0
        for other_idx, other_tile in enumerate(detected_matches):
            if i == other_idx:
                continue
            
            other_x = other_tile['local_x'] // tile_size
            other_y = other_tile['local_y'] // tile_size
            
            # Check if within +/- 2 tiles range
            dx = abs(tile_x - other_x)
            dy = abs(tile_y - other_y)
            
            if dx <= 2 and dy <= 2 and (dx + dy) > 0:  # (dx + dy) > 0 excludes self
                # Get or compute neighbor tile features
                if 'features' in other_tile:
                    neighbor_features = np.array(other_tile['features'])
                else:
                    neighbor_tile_img = img.crop((other_tile['local_x'], other_tile['local_y'],
                                                 other_tile['local_x'] + tile_size, other_tile['local_y'] + tile_size))
                    neighbor_features = compute_simple_tile_features(neighbor_tile_img)
                
                # Compute cosine similarity between reference and neighbor
                similarity = np.dot(ref_features, neighbor_features) / (
                    np.linalg.norm(ref_features) * np.linalg.norm(neighbor_features) + 1e-10
                )
                
                # Check if similarity is above threshold
                if similarity >= neighbor_threshold:
                    valid_neighbors += 1
        
        # # If not enough valid neighbors, mark for filtering
        # if valid_neighbors < min_neighbors:
        #     filtered_out_set.add(i)
    
    #logger.info(f"Filtering out {len(filtered_out_set)} isolated tiles (with < {min_neighbors} neighbors)")
    
    # Debug output: save colormap with threshold scores and filtered tiles
    if False: #is_debug:
        try:
            # Find next incrementing index
            debug_dir = Path("./debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            existing_debugs = list(debug_dir.glob("debug_traversability_colourmap_*.png"))
            next_index = 0
            if existing_debugs:
                indices = []
                for debug_file in existing_debugs:
                    try:
                        # Extract number from filename like "debug_traversability_colourmap_5.png"
                        filename = debug_file.stem
                        if "_colourmap_" in filename:
                            index_part = filename.split("_colourmap_")[-1]
                            indices.append(int(index_part))
                    except (ValueError, IndexError):
                        continue
                next_index = max(indices) + 1 if indices else 0
            
            # Create debug image with score overlays (active_tile_filenames already loaded above)
            debug_img = img.copy()
            draw = ImageDraw.Draw(debug_img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
            
            # Draw score overlays for all processed tiles
            match_idx = 0
            for cluster_id, matches in traversability_map.items():
                if cluster_id not in valid_cluster_ids:
                    continue
                for match in matches:
                    if match['score'] >= match_threshold:
                        x, y = match['local_x'], match['local_y']
                        score = match['score']
                        matched_filename = match.get('matched_filename', '')
                        
                        # Check if this tile is filtered out
                        is_filtered = match_idx in filtered_out_set
                        
                        # Check if this tile is from active cache
                        is_active_tile = matched_filename in active_tile_filenames
                        
                        # Draw score text
                        score_text = f"{score:.2f}"
                        if is_active_tile:
                            score_text += " [A]"  # Mark active tiles
                        
                        try:
                            bbox = draw.textbbox((x+2, y+2), score_text, font=font)
                            # Draw white background for text
                            draw.rectangle([bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1], fill='white')
                            # Use green text for active tiles, red for unique tiles
                            text_color = 'green' if is_active_tile else 'red'
                            draw.text((x+2, y+2), score_text, fill=text_color, font=font)
                        except:
                            text_color = 'green' if is_active_tile else 'red'
                            draw.text((x+2, y+2), score_text, fill=text_color, font=font)
                        
                        # Draw blue border for active tiles
                        if is_active_tile:
                            draw.rectangle([x, y, x+31, y+31], outline='blue', width=2)
                        
                        # Draw yellow cross below text for filtered tiles
                        if is_filtered:
                            cross_y_offset = 12  # Below the text
                            cross_size = 8
                            cross_center_x = x + 16
                            cross_center_y = y + cross_y_offset + 8
                            
                            # Draw yellow cross
                            draw.line([cross_center_x - cross_size, cross_center_y - cross_size,
                                      cross_center_x + cross_size, cross_center_y + cross_size],
                                     fill='yellow', width=2)
                            draw.line([cross_center_x - cross_size, cross_center_y + cross_size,
                                      cross_center_x + cross_size, cross_center_y - cross_size],
                                     fill='yellow', width=2)
                        
                        match_idx += 1
            
            # Save debug image
            debug_path = debug_dir / f"debug_traversability_colourmap_{next_index}.png"
            debug_img.save(debug_path)
            #logger.info(f"Saved debug colormap with scores for {match_class} tiles: {debug_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug colormap: {e}")
    
    # Remove filtered out tiles from detected_matches
    detected_matches = [tile for i, tile in enumerate(detected_matches) if i not in filtered_out_set]
    
    #logger.info(f"Detected {len(detected_matches)} tile matches in image (after filtering)")
    return detected_matches


def create_traversability_map_from_image(image_path: Optional[str] = None,
                                     output_dir: str = DEFAULT_TRAVERSABILITY_DIR,
                                     uid: Optional[str] = None,
                                     base_x: int = 0,
                                     base_y: int = 0,
                                     frame_pil: Optional[Image.Image] = None,
                                     map_name: Optional[str] = None,
                                     unique_tiles_dir: str = DEFAULT_UNIQUE_TILES_DIR,
                                     return_traversability_map: bool = False) -> Optional[Union[str, Tuple[str, Dict]]]:
    """
    Create a complete traversability map from an image.
    
    This is the main entry point that:
    1. Loads cluster and entity mappings
    2. Loads unique tiles data
    3. Builds traversability map from template matching
    4. Generates tilemap and identifies entities
    5. Saves all outputs
    
    Args:
        image_path: Path to input image or PIL Image
        output_dir: Output directory for traversability maps
        uid: Unique ID for this map
        base_x: Base tile x coordinate
        base_y: Base tile y coordinate
        frame_pil: Optional PIL Image (if image_path is None)
        map_name: Optional map name
        unique_tiles_dir: Directory containing unique tiles data
        return_traversability_map: If True, return (uid, traversability_map) instead of just uid
        
    Returns:
        uid if successful and return_traversability_map=False, (uid, traversability_map) if return_traversability_map=True, None otherwise
    """
    try:
        # Load image
        if frame_pil is None:
            frame_pil = Image.open(image_path)
        
        # upscale x2 if frame is too small
        if frame_pil.size < (480, 320):
            frame_pil = frame_pil.resize((480, 320), Image.NEAREST)
        
        # Pad the frame with 16px top and bottom only
        padded_height = frame_pil.height + 32
        padded_frame = Image.new('RGB', (frame_pil.width, padded_height), (0, 0, 0))
        padded_frame.paste(frame_pil, (0, 16))
        frame_pil = padded_frame
        
        # Load mappings
        entitymap_path = Path(unique_tiles_dir) / "entitymap.json"
        
        entity_mapping = {}
        if entitymap_path.exists():
            entity_mapping = load_entity_mapping(str(entitymap_path))
        else:
            logger.error(f"entitymap.json not found at {entitymap_path}")
            return None
        
        # Create synthetic cluster_mapping from entity_mapping + ENTITY_TYPES
        cluster_mapping = {}
        for cluster_id, entity_type_id in entity_mapping.items():
            entity_name = ENTITY_TYPES.get(entity_type_id, 'unknown')
            cluster_mapping[cluster_id] = entity_name
        
        # Load unique tiles
        processed_data = load_processed_tiles(unique_tiles_dir)
        if not processed_data:
            logger.error("No unique tiles data loaded")
            return None
        
        # Build traversability map
        tile_feature_cache = {}
        traversability_map = build_traversability_map_from_templates(
            frame_pil, base_x, base_y, processed_data, cluster_mapping, tile_feature_cache
        )
        
        # Generate tilemap
        tilemap_data = generate_traversability_tilemap(
            traversability_map, image_path, base_x, base_y, cluster_mapping
        )
        
        # Identify entities
        entities_data = identify_entities(traversability_map, entity_mapping, base_x, base_y)
        
        # Save outputs
        output_files = save_traversability_outputs(
            traversability_map, tilemap_data, entities_data, output_dir, uid,
            frame_pil, map_name, cluster_mapping, base_x, base_y
        )
        
        #logger.info(f"Created traversability map {uid}: {output_files}")
        if return_traversability_map:
            return uid, traversability_map
        else:
            return uid
        
    except Exception as e:
        logger.error(f"Error creating traversability map: {e}")
        return None
