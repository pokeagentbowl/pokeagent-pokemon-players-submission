"""Azure Computer Vision API wrapper for object detection in Pokemon game frames."""
import os
import time
import logging
from typing import List, Optional, Tuple, Dict, Literal
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class DetectedObject(BaseModel):
    """Structured data for detected objects from Azure Computer Vision API."""
    name: str
    confidence: float
    bbox: Dict[str, int]  # {x, y, w, h}
    center_pixel: Tuple[int, int]
    entity_type: Optional[str] = None
    source: Literal["object_detection", "dense_captioning", "people_detection", "uniquetiles_match"] = "object_detection"


class ObjectDetector:
    """Azure Computer Vision API wrapper for object detection."""
    
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None, 
                 save_debug_frames: bool = False, debug_output_dir: str = "debug_frames"):
        """
        Initialize Azure Computer Vision object detector.
        
        Args:
            endpoint: Azure Computer Vision endpoint (defaults to VISION_ENDPOINT env var)
            key: Azure Computer Vision API key (defaults to VISION_KEY env var)
            save_debug_frames: Whether to save annotated frames with bounding boxes
            debug_output_dir: Directory to save debug frames
        """
        self.endpoint = endpoint or os.environ.get("VISION_ENDPOINT")
        self.key = key or os.environ.get("VISION_KEY")
        
        # Fail hard if credentials not provided (no fallbacks)
        if not self.endpoint:
            raise ValueError("VISION_ENDPOINT environment variable not set")
        if not self.key:
            raise ValueError("VISION_KEY environment variable not set")
        
        # Initialize Azure client
        self.client = ImageAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        
        self.save_debug_frames = save_debug_frames
        self.debug_output_dir = debug_output_dir
        
        # Create debug output directory if needed
        if self.save_debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            logger.info(f"Debug frames will be saved to: {self.debug_output_dir}")
        
        logger.info(f"Initialized ObjectDetector with endpoint: {self.endpoint}")
    
    def detect_exact_tile_matches(self, frame: np.ndarray,
                                  match_threshold: float = 0.95, output_dir: Optional[str] = None, match_class:str = "npc") -> List[DetectedObject]:
        """
        Exact tiles matching logic ported from beliefs_NT/utils/label_traversable.py

        For now, we are running DIRECT detection per frame --> singular tile labels

        Future: Check frame --> traversability maps and retrieve from merged maps
        """
        from custom_utils.label_traversable import detect_tile_matches_from_image
        
        # Get matches from the traversable library
        detected_matches = detect_tile_matches_from_image(
            frame, match_threshold=match_threshold, match_class = match_class
        )
        
        # Convert matches to DetectedObject instances
        detected_objects = []
        for match in detected_matches:
            local_x = match['local_x']
            local_y = match['local_y']
            
            bbox = {
                'x': local_x,
                'y': local_y,
                'w': 32,
                'h': 32
            }
            
            center_x = local_x + 16
            center_y = local_y + 16
            
            detected_objects.append(DetectedObject(
                name=f"cluster_{match['cluster_id']}",
                confidence=match['score'],
                bbox=bbox,
                center_pixel=(center_x, center_y),
                entity_type="interactable",
                source="uniquetiles_match"
            ))
        
        logger.info(f"Detected {len(detected_objects)} exact tile matches in frame")
        return detected_objects


    def detect_objects(self, frame: np.ndarray, scale_factor: float = 4.0,
                      visual_features: Optional[List[VisualFeatures]] = None) -> List[DetectedObject]:
        """
        Detect objects in frame using Azure Computer Vision API.
        
        Args:
            frame: Numpy array representing the game frame (RGB, shape: [H, W, 3])
            scale_factor: Factor to scale the image before detection (default: 4.0)
                         Bounding boxes will be scaled back to original dimensions
            visual_features: Optional list of visual features to request. 
                           If None, uses all three: [OBJECTS, DENSE_CAPTIONS, PEOPLE]
            
        Returns:
            List of DetectedObject instances with bounding boxes and metadata
        """
        original_height, original_width = frame.shape[:2]
        
        # Scale frame if scale_factor != 1.0
        if scale_factor != 1.0:
            scaled_frame = self._scale_frame(frame, scale_factor)
            logger.info(f"Scaled frame from {original_width}x{original_height} to {scaled_frame.shape[1]}x{scaled_frame.shape[0]}")
        else:
            scaled_frame = frame
        
        # Get analysis result (dict format) - subclasses can override for caching
        result_dict = self._get_analysis_result(scaled_frame, visual_features)
        
        # Parse result dict into DetectedObject list
        detected_objects = self._parse_detection_response(result_dict, scale_factor)
        
        logger.info(f"Detected {len(detected_objects)} objects in frame (scale: {scale_factor}x)")
        
        # Save debug frame with bounding boxes if enabled
        if self.save_debug_frames and detected_objects:
            self._save_debug_frame(frame, detected_objects)
        
        return detected_objects
    
    def _get_analysis_result(self, scaled_frame: np.ndarray, 
                            visual_features: Optional[List[VisualFeatures]] = None) -> Dict:
        """
        Get analysis result from Azure API as dict.
        
        Subclasses can override this method to add caching logic.
        
        Args:
            scaled_frame: Scaled frame ready for API submission
            visual_features: Optional list of visual features to request
            
        Returns:
            Analysis result as dict (from result.as_dict())
        """
        # Default to all three features
        if visual_features is None:
            visual_features = [VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
        
        # Convert frame to PNG bytes for API submission
        image_data = self._frame_to_png_bytes(scaled_frame)
        
        # Call Azure Computer Vision API
        result = self.client.analyze(
            image_data=image_data,
            visual_features=visual_features
        )
        
        return result.as_dict()
    
    def _scale_frame(self, frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale frame by a given factor.
        
        Args:
            frame: Numpy array (RGB image)
            scale_factor: Factor to scale by
            
        Returns:
            Scaled numpy array
        """
        img = Image.fromarray(frame)
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
        return np.array(scaled_img)
    
    def _frame_to_png_bytes(self, frame: np.ndarray) -> bytes:
        """
        Convert numpy frame to PNG bytes for Azure API.
        
        Args:
            frame: Numpy array (RGB image)
            
        Returns:
            PNG encoded bytes
        """
        img = Image.fromarray(frame)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    def _parse_detection_response(self, result_dict: Dict, scale_factor: float = 1.0) -> List[DetectedObject]:
        """
        Parse Azure Computer Vision API response into DetectedObject list.
        
        Args:
            response: Azure ImageAnalysisResult object (will be converted to dict)
            scale_factor: Factor that was used to scale the image (for bbox scaling back)
            
        Returns:
            List of DetectedObject instances from all detected features
        """
        detected_objects = []
        detected_objects.extend(self._parse_objects(result_dict, scale_factor))
        detected_objects.extend(self._parse_dense_captions(result_dict, scale_factor))
        detected_objects.extend(self._parse_people(result_dict, scale_factor))
        return detected_objects
    
    def _parse_objects(self, data: Dict, scale_factor: float = 1.0) -> List[DetectedObject]:
        """
        Parse OBJECTS feature into DetectedObject list.
        
        Args:
            data: Dict with 'objectsResult' key (from result.as_dict())
            scale_factor: Factor to scale bounding boxes back to original dimensions
            
        Returns:
            List of DetectedObject instances from object detection
        """
        detected_objects = []
        
        objects_result = data.get('objectsResult', {})
        objects_list = objects_result.get('values', [])
        for obj_data in objects_list:
            if not obj_data.get('tags'):
                continue
            
            primary_tag = obj_data['tags'][0]
            bbox_data = obj_data['boundingBox']
            
            # Scale bbox back to original dimensions (240x160 if scale_factor=4.0)
            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }
            
            # Convert from 240x160 to 480x352 coordinate space
            # Reason: pixel_to_local_tile() expects padded frame coordinates (480x352)
            # but Azure CV receives 240x160 frame and returns bboxes in that space
            # Step 1: Scale 2x (240x160 -> 480x320)
            bbox['x'] *= 2
            bbox['y'] *= 2
            bbox['w'] *= 2
            bbox['h'] *= 2
            
            # Step 2: Add 16px top padding offset (480x320 -> 480x352)
            bbox['y'] += 16
            
            center_x = bbox['x'] + bbox['w'] // 2
            # For object detections, use bottom-center of bbox as anchor point
            # Reason: Objects/characters stand on tiles, so bottom of bbox determines tile position
            center_y = bbox['y'] + bbox['h']
            
            detected_objects.append(DetectedObject(
                name=primary_tag['name'],
                confidence=primary_tag['confidence'],
                bbox=bbox,
                center_pixel=(center_x, center_y),
                entity_type=None,
                source="object_detection"
            ))
        
        return detected_objects
    
    def _parse_dense_captions(self, data: Dict, scale_factor: float = 1.0) -> List[DetectedObject]:
        """
        Parse DENSE_CAPTIONS feature into DetectedObject list.
        
        Args:
            data: Dict with 'denseCaptionsResult' key (from result.as_dict())
            scale_factor: Factor to scale bounding boxes back to original dimensions
            
        Returns:
            List of DetectedObject instances from dense captioning
        """
        detected_objects = []
        
        captions_result = data.get('denseCaptionsResult', {})
        captions_list = captions_result.get('values', [])
        for caption_data in captions_list:
            bbox_data = caption_data['boundingBox']
            
            # Scale bbox back to original dimensions (240x160 if scale_factor=4.0)
            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }
            
            # Convert from 240x160 to 480x352 coordinate space
            # Reason: pixel_to_local_tile() expects padded frame coordinates (480x352)
            # but Azure CV receives 240x160 frame and returns bboxes in that space
            # Step 1: Scale 2x (240x160 -> 480x320)
            bbox['x'] *= 2
            bbox['y'] *= 2
            bbox['w'] *= 2
            bbox['h'] *= 2
            
            # Step 2: Add 16px top padding offset (480x320 -> 480x352)
            bbox['y'] += 16
            
            center_x = bbox['x'] + bbox['w'] // 2
            # For character/object detections, use bottom-center of bbox as anchor point
            # Reason: Characters stand on tiles, so bottom of bbox determines tile position
            center_y = bbox['y'] + bbox['h']
            
            detected_objects.append(DetectedObject(
                name=caption_data['text'],
                confidence=caption_data['confidence'],
                bbox=bbox,
                center_pixel=(center_x, center_y),
                entity_type=None,
                source="dense_captioning"
            ))
        
        return detected_objects
    
    def _parse_people(self, data: Dict, scale_factor: float = 1.0) -> List[DetectedObject]:
        """
        Parse PEOPLE feature into DetectedObject list.
        
        Args:
            data: Dict with 'peopleResult' key (from result.as_dict())
            scale_factor: Factor to scale bounding boxes back to original dimensions
            
        Returns:
            List of DetectedObject instances from people detection
        """
        detected_objects = []
        
        people_result = data.get('peopleResult', {})
        people_list = people_result.get('values', [])
        for person_data in people_list:
            bbox_data = person_data['boundingBox']
            
            # Scale bbox back to original dimensions (240x160 if scale_factor=4.0)
            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }
            
            # Convert from 240x160 to 480x352 coordinate space
            # Reason: pixel_to_local_tile() expects padded frame coordinates (480x352)
            # but Azure CV receives 240x160 frame and returns bboxes in that space
            # Step 1: Scale 2x (240x160 -> 480x320)
            bbox['x'] *= 2
            bbox['y'] *= 2
            bbox['w'] *= 2
            bbox['h'] *= 2
            
            # Step 2: Add 16px top padding offset (480x320 -> 480x352)
            bbox['y'] += 16
            
            center_x = bbox['x'] + bbox['w'] // 2
            # For people detections, use bottom-center of bbox as anchor point
            # Reason: People stand on tiles, so bottom of bbox determines tile position
            center_y = bbox['y'] + bbox['h']
            
            detected_objects.append(DetectedObject(
                name="person",
                confidence=person_data['confidence'],
                bbox=bbox,
                center_pixel=(center_x, center_y),
                entity_type=None,
                source="people_detection"
            ))
        
        return detected_objects
    
    def _save_debug_frame(self, frame: np.ndarray, detected_objects: List[DetectedObject]):
        """
        Save frame with bounding boxes, labels, and confidence scores drawn on it.
        
        Args:
            frame: Original numpy array frame
            detected_objects: List of detected objects with bounding boxes
        """
        # Convert frame to PIL Image for drawing
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        font = ImageFont.load_default()
        
        # Define colors for different sources
        source_colors = {
            "object_detection": "orange",
            "dense_captioning": "blue",
            "people_detection": "green"
        }
        
        # Draw bounding boxes and labels
        for obj in detected_objects:
            bbox = obj.bbox
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Get color based on source
            color = source_colors.get(obj.source, "red")
            
            # Draw rectangle
            draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=2)
            
            # Draw label with confidence and source prefix
            source_prefix = {"object_detection": "[OBJ]", "dense_captioning": "[DEN]", "people_detection": "[PPL]"}
            prefix = source_prefix.get(obj.source, "[UNK]")
            label = f"{prefix} {obj.name} ({obj.confidence:.2f})"
            
            # Draw text background for readability
            text_bbox = draw.textbbox((x, y - 12), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x, y - 12), label, fill="white", font=font)
        
        # Save with unix timestamp filename
        timestamp = int(time.time())
        filename = f"{timestamp}.png"
        filepath = os.path.join(self.debug_output_dir, filename)
        img.save(filepath)
        
        logger.debug(f"Saved debug frame to: {filepath}")

