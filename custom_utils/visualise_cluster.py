import json
from collections import defaultdict
import os
from PIL import Image, ImageDraw
from glob import glob

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

def visualise_cluster(entity_id):
    # Load entitymap.json
    with open('./uniqueembeddings/entitymap.json', 'r') as f:
        entitymap = json.load(f)

    unique_tiles = glob('./uniqueembeddings/*.png')


    # Group tiles by cluster for the given entity_id
    cluster_to_tiles = defaultdict(list)
    for tile in unique_tiles:
        if "debug" in tile or "montage" in tile or "centroids" in tile:
            continue
        filestem = os.path.splitext(os.path.basename(tile))[0]
        cluster, uid = filestem.split("_")
        if cluster in entitymap and entitymap[cluster] == entity_id:
            cluster_to_tiles[cluster].append((tile, uid))

    if not cluster_to_tiles:
        print(f"No tiles found for entity_id {entity_id}")
        return
    
    # Sort clusters and tiles within each cluster by uid
    sorted_clusters = sorted(cluster_to_tiles.items())
    
    TILE_SIZE = 32
    TILES_PER_ROW = 10
    HEADER_HEIGHT = 20
    
    # Calculate dimensions
    total_rows = 0
    for cluster, tiles in sorted_clusters:
        # Add header row
        total_rows += 1
        # Add tile rows for this cluster
        num_tiles = len(tiles)
        tile_rows = (num_tiles + TILES_PER_ROW - 1) // TILES_PER_ROW
        total_rows += tile_rows
    
    # Calculate montage dimensions
    montage_width = TILES_PER_ROW * TILE_SIZE
    montage_height = 0
    for cluster, tiles in sorted_clusters:
        montage_height += HEADER_HEIGHT  # Header row
        num_tiles = len(tiles)
        tile_rows = (num_tiles + TILES_PER_ROW - 1) // TILES_PER_ROW
        montage_height += tile_rows * TILE_SIZE
    
    # Create montage image
    montage = Image.new('RGB', (montage_width, montage_height), color=(255, 255, 255))
    montage_draw = ImageDraw.Draw(montage)
    
    y_offset = 0
    for cluster, tiles in sorted_clusters:
        # Draw cluster header
        montage_draw.text((5, y_offset + 2), f"Cluster {cluster}", fill=(0, 0, 0))
        y_offset += HEADER_HEIGHT
        
        # Sort tiles by uid
        sorted_tiles = sorted(tiles, key=lambda x: int(x[1]))
        
        # Draw tiles
        for idx, (fname, uid) in enumerate(sorted_tiles):
            row = idx // TILES_PER_ROW
            col = idx % TILES_PER_ROW
            
            x_pos = col * TILE_SIZE
            y_pos = y_offset + row * TILE_SIZE
            
            # Load and paste tile
            img = Image.open(fname)
            montage.paste(img, (x_pos, y_pos))
            
            # Draw UID on tile with a white background
            from PIL import ImageFont

            uid_text = str(uid)
            font = ImageFont.load_default()
            bbox = montage_draw.textbbox((x_pos + 2, y_pos + 2), uid_text, font=font)
            montage_draw.rectangle(bbox, fill=(255, 255, 255))
            montage_draw.text((x_pos + 2, y_pos + 2), uid_text, fill=(0, 0, 0), font=font)
        
        # Move to next cluster
        num_tiles = len(sorted_tiles)
        tile_rows = (num_tiles + TILES_PER_ROW - 1) // TILES_PER_ROW
        y_offset += tile_rows * TILE_SIZE
    
    # Save montage
    entity_name = ENTITY_TYPES.get(entity_id, f"entity_{entity_id}")
    output_path = f'./montage_{entity_name}_entity{entity_id}.png'
    montage.save(output_path)
    print(f"Saved montage: {output_path}")

# Example usage
if __name__ == "__main__":
    # Visualise grass clusters (entity_id 7)
    visualise_cluster(2)
