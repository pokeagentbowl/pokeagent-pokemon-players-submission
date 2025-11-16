#!/usr/bin/env python3
"""
Unit tests for CachedObjectDetector

Tests the caching functionality for Azure Computer Vision API wrapper.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from custom_utils.cached_object_detector import CachedObjectDetector
from custom_utils.object_detector import DetectedObject
from azure.ai.vision.imageanalysis.models import VisualFeatures


class MockBoundingBox:
    """Mock Azure BoundingBox object."""
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class MockTag:
    """Mock Azure Tag object."""
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence


class MockObject:
    """Mock Azure detected object."""
    def __init__(self, name, confidence, x, y, w, h):
        self.tags = [MockTag(name, confidence)]
        self.bounding_box = MockBoundingBox(x, y, w, h)


class MockCaption:
    """Mock Azure dense caption."""
    def __init__(self, text, confidence, x, y, w, h):
        self.text = text
        self.confidence = confidence
        self.bounding_box = MockBoundingBox(x, y, w, h)


class MockPerson:
    """Mock Azure person detection."""
    def __init__(self, confidence, x, y, w, h):
        self.confidence = confidence
        self.bounding_box = MockBoundingBox(x, y, w, h)


class MockAnalysisResult:
    """Mock Azure analysis result."""
    def __init__(self, objects=None, dense_captions=None, people=None):
        self.objects = Mock(list=objects) if objects else None
        self.dense_captions = Mock(list=dense_captions) if dense_captions else None
        self.people = Mock(list=people) if people else None


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_cache_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return np.zeros((160, 240, 3), dtype=np.uint8)


@pytest.fixture
def mock_azure_client():
    """Create a mock Azure client."""
    with patch('custom_utils.cached_object_detector.ImageAnalysisClient') as mock:
        yield mock


class TestCachedObjectDetector:
    """Test suite for CachedObjectDetector."""
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_initialization(self, temp_cache_dir):
        """Test that detector initializes correctly."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        assert detector.cache is not None
        assert Path(temp_cache_dir).exists()
        assert os.path.exists(os.path.join(temp_cache_dir, "vision_cache.db"))
    
    def test_image_hash_generation(self, temp_cache_dir, test_image):
        """Test that image hash is generated consistently."""
        with patch.dict(os.environ, {'VISION_ENDPOINT': 'test', 'VISION_KEY': 'test'}):
            detector = CachedObjectDetector(cache_dir=temp_cache_dir)
            
            hash1 = detector._generate_image_hash(test_image)
            hash2 = detector._generate_image_hash(test_image)
            
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 produces 64 hex characters
    
    def test_image_hash_uniqueness(self, temp_cache_dir):
        """Test that different images produce different hashes."""
        with patch.dict(os.environ, {'VISION_ENDPOINT': 'test', 'VISION_KEY': 'test'}):
            detector = CachedObjectDetector(cache_dir=temp_cache_dir)
            
            image1 = np.zeros((160, 240, 3), dtype=np.uint8)
            image2 = np.ones((160, 240, 3), dtype=np.uint8)
            
            hash1 = detector._generate_image_hash(image1)
            hash2 = detector._generate_image_hash(image2)
            
            assert hash1 != hash2
    
    def test_cache_key_generation(self, temp_cache_dir):
        """Test cache key generation for different features."""
        with patch.dict(os.environ, {'VISION_ENDPOINT': 'test', 'VISION_KEY': 'test'}):
            detector = CachedObjectDetector(cache_dir=temp_cache_dir)
            
            key1 = detector._generate_cache_key("test_hash", "OBJECTS")
            key2 = detector._generate_cache_key("test_hash", "DENSE_CAPTIONS")
            key3 = detector._generate_cache_key("different_hash", "OBJECTS")
            
            assert key1 != key2  # Different features
            assert key1 != key3  # Different hashes
            assert "vision_" in key1
            assert "test_hash" in key1
            assert "OBJECTS" in key1
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_feature_data_extraction_objects(self, temp_cache_dir):
        """Test extraction of object detection data."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        # Create mock result
        mock_result = MockAnalysisResult(
            objects=[MockObject("cat", 0.9, 10, 20, 30, 40)]
        )
        
        feature_data = detector._extract_feature_data(mock_result, VisualFeatures.OBJECTS)
        
        assert "objects" in feature_data
        assert len(feature_data["objects"]) == 1
        assert feature_data["objects"][0]["tags"][0]["name"] == "cat"
        assert feature_data["objects"][0]["tags"][0]["confidence"] == 0.9
        assert feature_data["objects"][0]["bounding_box"]["x"] == 10
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_feature_data_extraction_dense_captions(self, temp_cache_dir):
        """Test extraction of dense captions data."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        mock_result = MockAnalysisResult(
            dense_captions=[MockCaption("a cat sitting", 0.85, 5, 10, 50, 60)]
        )
        
        feature_data = detector._extract_feature_data(mock_result, VisualFeatures.DENSE_CAPTIONS)
        
        assert "dense_captions" in feature_data
        assert len(feature_data["dense_captions"]) == 1
        assert feature_data["dense_captions"][0]["text"] == "a cat sitting"
        assert feature_data["dense_captions"][0]["confidence"] == 0.85
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_feature_data_extraction_people(self, temp_cache_dir):
        """Test extraction of people detection data."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        mock_result = MockAnalysisResult(
            people=[MockPerson(0.95, 15, 25, 35, 45)]
        )
        
        feature_data = detector._extract_feature_data(mock_result, VisualFeatures.PEOPLE)
        
        assert "people" in feature_data
        assert len(feature_data["people"]) == 1
        assert feature_data["people"][0]["confidence"] == 0.95
        assert feature_data["people"][0]["bounding_box"]["x"] == 15
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_parse_cached_results_objects(self, temp_cache_dir):
        """Test parsing of cached object detection results."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.OBJECTS: {
                "objects": [{
                    "tags": [{"name": "dog", "confidence": 0.88}],
                    "bounding_box": {"x": 100, "y": 100, "width": 50, "height": 50}
                }]
            }
        }
        
        results = detector._parse_cached_results(cached_data, scale_factor=1.0)
        
        assert len(results) == 1
        assert results[0].name == "dog"
        assert results[0].confidence == 0.88
        assert results[0].source == "object_detection"
        assert results[0].bbox == {'x': 100, 'y': 100, 'w': 50, 'h': 50}
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_parse_cached_results_dense_captions(self, temp_cache_dir):
        """Test parsing of cached dense captions results."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.DENSE_CAPTIONS: {
                "dense_captions": [{
                    "text": "a brown dog",
                    "confidence": 0.92,
                    "bounding_box": {"x": 50, "y": 60, "width": 70, "height": 80}
                }]
            }
        }
        
        results = detector._parse_cached_results(cached_data, scale_factor=1.0)
        
        assert len(results) == 1
        assert results[0].name == "a brown dog"
        assert results[0].confidence == 0.92
        assert results[0].source == "dense_captioning"
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_parse_cached_results_people(self, temp_cache_dir):
        """Test parsing of cached people detection results."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.PEOPLE: {
                "people": [{
                    "confidence": 0.97,
                    "bounding_box": {"x": 25, "y": 35, "width": 45, "height": 55}
                }]
            }
        }
        
        results = detector._parse_cached_results(cached_data, scale_factor=1.0)
        
        assert len(results) == 1
        assert results[0].name == "person"
        assert results[0].confidence == 0.97
        assert results[0].source == "people_detection"
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_parse_cached_results_multiple_features(self, temp_cache_dir):
        """Test parsing results from multiple features."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.OBJECTS: {
                "objects": [{
                    "tags": [{"name": "chair", "confidence": 0.75}],
                    "bounding_box": {"x": 10, "y": 20, "width": 30, "height": 40}
                }]
            },
            VisualFeatures.PEOPLE: {
                "people": [{
                    "confidence": 0.89,
                    "bounding_box": {"x": 50, "y": 60, "width": 70, "height": 80}
                }]
            }
        }
        
        results = detector._parse_cached_results(cached_data, scale_factor=1.0)
        
        assert len(results) == 2
        sources = {r.source for r in results}
        assert "object_detection" in sources
        assert "people_detection" in sources
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_scale_factor_applied_correctly(self, temp_cache_dir):
        """Test that scale factor is applied to bounding boxes."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.OBJECTS: {
                "objects": [{
                    "tags": [{"name": "ball", "confidence": 0.82}],
                    "bounding_box": {"x": 200, "y": 200, "width": 100, "height": 100}
                }]
            }
        }
        
        # Parse with scale factor of 2.0 (should divide bbox coordinates by 2)
        results = detector._parse_cached_results(cached_data, scale_factor=2.0)
        
        assert len(results) == 1
        assert results[0].bbox == {'x': 100, 'y': 100, 'w': 50, 'h': 50}
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_center_pixel_calculated_correctly(self, temp_cache_dir):
        """Test that center pixel is calculated correctly."""
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        cached_data = {
            VisualFeatures.OBJECTS: {
                "objects": [{
                    "tags": [{"name": "box", "confidence": 0.80}],
                    "bounding_box": {"x": 10, "y": 20, "width": 40, "height": 60}
                }]
            }
        }
        
        results = detector._parse_cached_results(cached_data, scale_factor=1.0)
        
        assert len(results) == 1
        # Center should be (10 + 40//2, 20 + 60//2) = (30, 50)
        assert results[0].center_pixel == (30, 50)
    
    @pytest.mark.skipif(
        not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"),
        reason="Azure credentials not configured"
    )
    def test_drop_in_replacement_compatibility(self, temp_cache_dir, test_image):
        """Test that CachedObjectDetector can be used as drop-in replacement for ObjectDetector."""
        # Initialize detector without visual_features parameter
        detector = CachedObjectDetector(cache_dir=temp_cache_dir)
        
        # Mock the Azure client to avoid actual API calls
        mock_result = MockAnalysisResult(
            objects=[MockObject("test", 0.9, 10, 20, 30, 40)]
        )
        
        with patch.object(detector.client, 'analyze', return_value=mock_result):
            # Call detect_objects without visual_features parameter
            # This should work just like the parent ObjectDetector class
            results = detector.detect_objects(test_image, scale_factor=1.0)
            
            # Verify the call was made with default features
            detector.client.analyze.assert_called_once()
            call_args = detector.client.analyze.call_args
            
            # Should have been called with all three default features
            features = call_args.kwargs['visual_features']
            assert VisualFeatures.OBJECTS in features
            assert VisualFeatures.DENSE_CAPTIONS in features
            assert VisualFeatures.PEOPLE in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
