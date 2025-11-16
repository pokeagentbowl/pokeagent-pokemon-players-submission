"""Azure Computer Vision API wrapper for comprehensive vision analysis (objects + OCR)."""
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

from custom_utils.object_detector_nt import DetectedObject


logger = logging.getLogger(__name__)


# class DetectedObject(BaseModel):
#     """Structured data for detected objects from Azure Computer Vision API."""
#     name: str
#     confidence: float
#     bbox: Dict[str, int]  # {x, y, w, h}
#     center_pixel: Tuple[int, int]
#     entity_type: Optional[str] = None
#     source: Literal["object_detection", "dense_captioning", "people_detection", "uniquetiles_match"] = "object_detection"

class OCRTextBlock(BaseModel):
    """Structured data for OCR text from Azure Computer Vision READ API."""
    text: str
    confidence: float
    bounding_polygon: List[Tuple[int, int]]  # [(x, y), ...]
    word_level: bool = False  # True if word, False if line


class VisionAnalysisResult(BaseModel):
    """Combined result from Azure Computer Vision analysis."""
    detected_objects: List[DetectedObject]
    ocr_text: List[OCRTextBlock]

    class Config:
        arbitrary_types_allowed = True


class AzureVisionAnalyzer:
    """Azure Computer Vision API wrapper for comprehensive vision analysis (objects + OCR)."""

    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None,
                 save_debug_frames: bool = False, debug_output_dir: str = "debug_frames"):
        """
        Initialize Azure Computer Vision analyzer.

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

        logger.info(f"Initialized AzureVisionAnalyzer with endpoint: {self.endpoint}")

    def analyze(self, frame: np.ndarray, scale_factor: float = 4.0,
                visual_features: Optional[List[VisualFeatures]] = None) -> VisionAnalysisResult:
        """
        Analyze frame using Azure Computer Vision API.

        Args:
            frame: Numpy array representing the game frame (RGB, shape: [H, W, 3])
            scale_factor: Factor to scale the image before detection (default: 4.0)
                         Bounding boxes will be scaled back to original dimensions
            visual_features: Optional list of visual features to request.
                           If None, uses all four: [OBJECTS, DENSE_CAPTIONS, PEOPLE, READ]

        Returns:
            VisionAnalysisResult with detected objects and OCR text
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

        # Parse result dict into structured outputs
        detected_objects = self._parse_detection_response(result_dict, scale_factor)
        ocr_text = self._parse_ocr_response(result_dict, scale_factor)

        logger.info(f"Analysis complete: {len(detected_objects)} objects, {len(ocr_text)} text blocks")

        # Save debug frame with bounding boxes if enabled
        if self.save_debug_frames and (detected_objects or ocr_text):
            self._save_debug_frame(frame, detected_objects, ocr_text)

        return VisionAnalysisResult(
            detected_objects=detected_objects,
            ocr_text=ocr_text
        )

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
        # Default to all four features (objects + OCR)
        if visual_features is None:
            visual_features = [
                VisualFeatures.OBJECTS,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.PEOPLE,
                VisualFeatures.READ
            ]

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
            result_dict: Azure ImageAnalysisResult dict (from result.as_dict())
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

            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }

            center_x = bbox['x'] + bbox['w'] // 2
            center_y = bbox['y'] + bbox['h'] // 2

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

            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }

            center_x = bbox['x'] + bbox['w'] // 2
            center_y = bbox['y'] + bbox['h'] // 2

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

            bbox = {
                'x': round(bbox_data['x'] / scale_factor),
                'y': round(bbox_data['y'] / scale_factor),
                'w': round(bbox_data['w'] / scale_factor),
                'h': round(bbox_data['h'] / scale_factor)
            }

            center_x = bbox['x'] + bbox['w'] // 2
            center_y = bbox['y'] + bbox['h'] // 2

            detected_objects.append(DetectedObject(
                name="person",
                confidence=person_data['confidence'],
                bbox=bbox,
                center_pixel=(center_x, center_y),
                entity_type=None,
                source="people_detection"
            ))

        return detected_objects

    def _parse_ocr_response(self, result_dict: Dict, scale_factor: float = 1.0) -> List[OCRTextBlock]:
        """
        Parse READ feature into OCRTextBlock list.

        Args:
            result_dict: Dict with 'readResult' key (from result.as_dict())
            scale_factor: Factor to scale bounding polygons back to original dimensions

        Returns:
            List of OCRTextBlock instances from OCR (both line-level and word-level)
        """
        ocr_blocks = []

        read_result = result_dict.get('readResult', {})
        if not read_result:
            return ocr_blocks

        blocks = read_result.get('blocks', [])
        for block in blocks:
            lines = block.get('lines', [])
            for line in lines:
                # Add line-level text block
                line_text = line.get('text', '')
                line_polygon = line.get('boundingPolygon', [])

                if line_polygon:
                    # Scale polygon coordinates back to original dimensions
                    scaled_polygon = [
                        (round(pt['x'] / scale_factor), round(pt['y'] / scale_factor))
                        for pt in line_polygon
                    ]

                    # Line confidence is average of word confidences
                    words = line.get('words', [])
                    word_confidences = [w.get('confidence', 0.0) for w in words]
                    line_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0

                    ocr_blocks.append(OCRTextBlock(
                        text=line_text,
                        confidence=line_confidence,
                        bounding_polygon=scaled_polygon,
                        word_level=False
                    ))

                # Add word-level text blocks
                for word in words:
                    word_text = word.get('text', '')
                    word_confidence = word.get('confidence', 0.0)
                    word_polygon = word.get('boundingPolygon', [])

                    if word_polygon:
                        scaled_word_polygon = [
                            (round(pt['x'] / scale_factor), round(pt['y'] / scale_factor))
                            for pt in word_polygon
                        ]

                        ocr_blocks.append(OCRTextBlock(
                            text=word_text,
                            confidence=word_confidence,
                            bounding_polygon=scaled_word_polygon,
                            word_level=True
                        ))

        return ocr_blocks

    def _save_debug_frame(self, frame: np.ndarray, detected_objects: List[DetectedObject],
                         ocr_text: List[OCRTextBlock]):
        """
        Save frame with bounding boxes for both objects and OCR text.

        Args:
            frame: Original numpy array frame
            detected_objects: List of detected objects with bounding boxes
            ocr_text: List of OCR text blocks with bounding polygons
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

        # Draw object detection bounding boxes
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

        # Draw OCR text polygons
        for ocr_block in ocr_text:
            # Use different colors for lines vs words
            color = "cyan" if not ocr_block.word_level else "yellow"

            # Draw polygon
            if len(ocr_block.bounding_polygon) >= 3:
                draw.polygon(ocr_block.bounding_polygon, outline=color, width=2 if not ocr_block.word_level else 1)

                # Draw text label for lines only (to avoid clutter)
                if not ocr_block.word_level:
                    x, y = ocr_block.bounding_polygon[0]
                    label = f"[OCR] {ocr_block.text[:20]} ({ocr_block.confidence:.2f})"
                    text_bbox = draw.textbbox((x, y - 12), label, font=font)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x, y - 12), label, fill="black", font=font)

        # Save with unix timestamp filename
        timestamp = int(time.time())
        filename = f"{timestamp}.png"
        filepath = os.path.join(self.debug_output_dir, filename)
        img.save(filepath)

        logger.debug(f"Saved debug frame to: {filepath}")
