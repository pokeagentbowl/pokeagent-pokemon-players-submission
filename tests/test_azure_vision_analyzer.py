"""Tests for AzureVisionAnalyzer - comprehensive object detection + OCR."""
import os
import numpy as np
import pytest
from PIL import Image

from custom_utils.azure_vision_analyzer import (
    AzureVisionAnalyzer,
    VisionAnalysisResult,
    DetectedObject,
    OCRTextBlock
)
from custom_utils.cached_azure_vision_analyzer import CachedAzureVisionAnalyzer
from azure.ai.vision.imageanalysis.models import VisualFeatures


@pytest.fixture
def analyzer():
    """Create AzureVisionAnalyzer instance (requires credentials)."""
    # Skip test if credentials not available
    if not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"):
        pytest.skip("VISION_ENDPOINT and VISION_KEY environment variables required")

    return AzureVisionAnalyzer()


@pytest.fixture
def cached_analyzer():
    """Create CachedAzureVisionAnalyzer instance (requires credentials)."""
    # Skip test if credentials not available
    if not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"):
        pytest.skip("VISION_ENDPOINT and VISION_KEY environment variables required")

    return CachedAzureVisionAnalyzer(prefer_redis=False)  # Use local file cache for testing


@pytest.fixture
def baseline_frame():
    """Load baseline screenshot from fixtures."""
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "baseline_screenshot.png"
    )

    if not os.path.exists(fixture_path):
        pytest.skip(f"Baseline screenshot not found at {fixture_path}")

    img = Image.open(fixture_path)
    return np.array(img)


@pytest.fixture
def menu_frame():
    """Load menu screenshot from fixtures."""
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "menu_screenshot.png"
    )

    if not os.path.exists(fixture_path):
        pytest.skip(f"Menu screenshot not found at {fixture_path}")

    img = Image.open(fixture_path)
    return np.array(img)


def test_analyzer_initialization():
    """Test analyzer can be initialized without credentials if skipped."""
    # This test checks that the analyzer properly validates credentials
    with pytest.raises(ValueError, match="VISION_ENDPOINT"):
        AzureVisionAnalyzer(endpoint=None, key="dummy")

    with pytest.raises(ValueError, match="VISION_KEY"):
        AzureVisionAnalyzer(endpoint="dummy", key=None)


def test_baseline_analysis(analyzer, baseline_frame):
    """Test analysis on baseline screenshot (no text expected)."""
    result = analyzer.analyze(baseline_frame, scale_factor=4.0)

    # Should return VisionAnalysisResult
    assert isinstance(result, VisionAnalysisResult)

    # Should have detected_objects and ocr_text lists
    assert isinstance(result.detected_objects, list)
    assert isinstance(result.ocr_text, list)

    # Baseline typically has no OCR text
    assert len(result.ocr_text) == 0 or all(isinstance(t, OCRTextBlock) for t in result.ocr_text)

    # May or may not have objects, but should be valid DetectedObjects
    if result.detected_objects:
        assert all(isinstance(obj, DetectedObject) for obj in result.detected_objects)


def test_menu_analysis(analyzer, menu_frame):
    """Test analysis on menu screenshot (text expected)."""
    result = analyzer.analyze(menu_frame, scale_factor=4.0)

    # Should return VisionAnalysisResult
    assert isinstance(result, VisionAnalysisResult)

    # Menu typically has OCR text
    assert len(result.ocr_text) > 0
    assert all(isinstance(t, OCRTextBlock) for t in result.ocr_text)

    # Check OCR text structure
    for ocr_block in result.ocr_text:
        assert isinstance(ocr_block.text, str)
        assert 0.0 <= ocr_block.confidence <= 1.0
        assert len(ocr_block.bounding_polygon) >= 3  # Polygon has at least 3 points
        assert isinstance(ocr_block.word_level, bool)


def test_selective_features(analyzer, baseline_frame):
    """Test requesting only specific features."""
    # Request only READ feature
    result = analyzer.analyze(
        baseline_frame,
        scale_factor=4.0,
        visual_features=[VisualFeatures.READ]
    )

    # Should still return VisionAnalysisResult
    assert isinstance(result, VisionAnalysisResult)

    # Should have no detected objects (we didn't request OBJECTS/DENSE_CAPTIONS/PEOPLE)
    assert len(result.detected_objects) == 0

    # May or may not have OCR text depending on image content
    assert isinstance(result.ocr_text, list)


def test_cached_analyzer(cached_analyzer, baseline_frame):
    """Test cached analyzer caches results properly."""
    # First call - should hit API
    result1 = cached_analyzer.analyze(baseline_frame, scale_factor=4.0)

    # Second call - should use cache (check logs for "Cache HIT")
    result2 = cached_analyzer.analyze(baseline_frame, scale_factor=4.0)

    # Results should be identical
    assert len(result1.detected_objects) == len(result2.detected_objects)
    assert len(result1.ocr_text) == len(result2.ocr_text)

    # Verify structure
    assert isinstance(result2, VisionAnalysisResult)


def test_cache_namespace_compatibility(cached_analyzer, baseline_frame):
    """Test that cache keys follow the same namespace as CachedObjectDetector."""
    # Generate cache key to verify format
    image_hash = cached_analyzer._generate_image_hash(baseline_frame)

    # Check that cache keys follow vision_{hash}_{feature} format
    for feature in ["OBJECTS", "DENSE_CAPTIONS", "PEOPLE", "READ"]:
        cache_key = cached_analyzer._generate_cache_key(image_hash, feature)
        assert cache_key == f"vision_{image_hash}_{feature}"

        # Cache key should be consistent with CachedObjectDetector naming
        assert cache_key.startswith("vision_")


def test_ocr_text_block_validation():
    """Test OCRTextBlock model validation."""
    # Valid OCR block
    ocr_block = OCRTextBlock(
        text="POKEMON",
        confidence=0.95,
        bounding_polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        word_level=True
    )

    assert ocr_block.text == "POKEMON"
    assert ocr_block.confidence == 0.95
    assert len(ocr_block.bounding_polygon) == 4
    assert ocr_block.word_level is True


def test_detected_object_validation():
    """Test DetectedObject model validation."""
    # Valid detected object
    obj = DetectedObject(
        name="pikachu",
        confidence=0.85,
        bbox={'x': 10, 'y': 20, 'w': 50, 'h': 50},
        center_pixel=(35, 45),
        source="object_detection"
    )

    assert obj.name == "pikachu"
    assert obj.confidence == 0.85
    assert obj.bbox == {'x': 10, 'y': 20, 'w': 50, 'h': 50}
    assert obj.center_pixel == (35, 45)
    assert obj.source == "object_detection"
