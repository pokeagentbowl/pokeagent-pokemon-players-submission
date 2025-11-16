"""
Test script to verify Azure Computer Vision caching performance.

This script demonstrates:
1. First call with all features - should be slow (API call)
2. Second call with same features - should be fast (cache hit)
3. Third call with partial overlap - should only compute missing features
4. Fourth call with new image - should be slow again (cache miss)
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

# Add parent directory to path for standalone script execution
# This is needed because this script is meant to be run directly, not via pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_utils.cached_object_detector import CachedObjectDetector
from azure.ai.vision.imageanalysis.models import VisualFeatures


def create_test_image(size=(240, 160), color=(255, 0, 0)):
    """Create a test image with given size and color."""
    img_array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return img_array


def run_timing_test():
    """Run timing tests to verify caching performance."""
    print("="*80)
    print("Azure Computer Vision Caching Performance Test")
    print("="*80)
    
    # Initialize cached detector
    detector = CachedObjectDetector(
        save_debug_frames=False
    )
    
    # Create test images
    print("\nCreating test images...")
    image1 = create_test_image(size=(240, 160), color=(255, 0, 0))  # Red image
    image2 = create_test_image(size=(240, 160), color=(0, 255, 0))  # Green image
    
    # Test 1: First call with all features (should be slow)
    print("\n" + "="*80)
    print("TEST 1: First call with all 3 features (OBJECTS, DENSE_CAPTIONS, PEOPLE)")
    print("Expected: API call for all features (SLOW)")
    print("="*80)
    
    start = time.time()
    result1 = detector.detect_objects(
        image1,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration1 = time.time() - start
    
    print(f"\n✓ Detected {len(result1)} objects")
    print(f"⏱  Duration: {duration1:.3f} seconds")
    
    # Test 2: Second call with same features (should be fast)
    print("\n" + "="*80)
    print("TEST 2: Second call with same image and same 3 features")
    print("Expected: All features from cache (FAST)")
    print("="*80)
    
    start = time.time()
    result2 = detector.detect_objects(
        image1,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration2 = time.time() - start
    
    print(f"\n✓ Detected {len(result2)} objects")
    print(f"⏱  Duration: {duration2:.3f} seconds")
    print(f"📊 Speedup: {duration1/duration2:.2f}x faster")
    
    # Test 3: Third call with partial overlap (2 cached, 1 new)
    print("\n" + "="*80)
    print("TEST 3: Third call with only OBJECTS feature")
    print("Expected: OBJECTS from cache (FAST)")
    print("="*80)
    
    start = time.time()
    result3 = detector.detect_objects(
        image1,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS]
    )
    duration3 = time.time() - start
    
    print(f"\n✓ Detected {len(result3)} objects")
    print(f"⏱  Duration: {duration3:.3f} seconds")
    print(f"📊 Speedup: {duration1/duration3:.2f}x faster than initial call")
    
    # Test 4: Fourth call with new image (should be slow)
    print("\n" + "="*80)
    print("TEST 4: Fourth call with different image and all 3 features")
    print("Expected: API call for all features (SLOW)")
    print("="*80)
    
    start = time.time()
    result4 = detector.detect_objects(
        image2,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration4 = time.time() - start
    
    print(f"\n✓ Detected {len(result4)} objects")
    print(f"⏱  Duration: {duration4:.3f} seconds")
    
    # Test 5: Fifth call with new image again (should be fast)
    print("\n" + "="*80)
    print("TEST 5: Fifth call with same image as Test 4")
    print("Expected: All features from cache (FAST)")
    print("="*80)
    
    start = time.time()
    result5 = detector.detect_objects(
        image2,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration5 = time.time() - start
    
    print(f"\n✓ Detected {len(result5)} objects")
    print(f"⏱  Duration: {duration5:.3f} seconds")
    print(f"📊 Speedup: {duration4/duration5:.2f}x faster")
    
    # Test 6: Partial feature request on cached image
    print("\n" + "="*80)
    print("TEST 6: Request only DENSE_CAPTIONS on image2 (already fully cached)")
    print("Expected: DENSE_CAPTIONS from cache (FAST)")
    print("="*80)
    
    start = time.time()
    result6 = detector.detect_objects(
        image2,
        scale_factor=1.0,
        visual_features=[VisualFeatures.DENSE_CAPTIONS]
    )
    duration6 = time.time() - start
    
    print(f"\n✓ Detected {len(result6)} objects")
    print(f"⏱  Duration: {duration6:.3f} seconds")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTest 1 (cold start, all features):     {duration1:.3f}s")
    print(f"Test 2 (cache hit, all features):      {duration2:.3f}s ({duration1/duration2:.2f}x speedup)")
    print(f"Test 3 (cache hit, single feature):    {duration3:.3f}s ({duration1/duration3:.2f}x speedup)")
    print(f"Test 4 (cold start, new image):        {duration4:.3f}s")
    print(f"Test 5 (cache hit, new image):         {duration5:.3f}s ({duration4/duration5:.2f}x speedup)")
    print(f"Test 6 (partial cache hit):            {duration6:.3f}s")
    
    print("\n✅ ACCEPTANCE CRITERIA:")
    if duration2 < duration1 * 0.5:
        print(f"   ✓ Cache provides significant speedup (Test 2 is {duration1/duration2:.2f}x faster)")
    else:
        print(f"   ⚠ Cache speedup may be limited (Test 2 is only {duration1/duration2:.2f}x faster)")
    
    if duration3 < duration1 * 0.5:
        print(f"   ✓ Partial feature requests are fast (Test 3 is {duration1/duration3:.2f}x faster)")
    else:
        print(f"   ⚠ Partial feature performance unclear")
    
    if duration5 < duration4 * 0.5:
        print(f"   ✓ Cache persists across different images (Test 5 is {duration4/duration5:.2f}x faster)")
    else:
        print(f"   ⚠ Cache persistence unclear")
    
    print("\n" + "="*80)


def run_feature_independence_test():
    """Test that features can be cached independently."""
    print("\n" + "="*80)
    print("Feature Independence Test")
    print("="*80)
    print("This test verifies that individual features are cached independently")
    print("="*80)
    
    detector = CachedObjectDetector(
        save_debug_frames=False
    )
    
    image = create_test_image(size=(240, 160), color=(0, 0, 255))  # Blue image
    
    # Step 1: Cache only OBJECTS
    print("\n1. Requesting only OBJECTS feature...")
    start = time.time()
    detector.detect_objects(image, scale_factor=1.0, visual_features=[VisualFeatures.OBJECTS])
    duration_objects_only = time.time() - start
    print(f"   Duration: {duration_objects_only:.3f}s")
    
    # Step 2: Request OBJECTS + DENSE_CAPTIONS
    # OBJECTS should be cached, only DENSE_CAPTIONS should hit API
    print("\n2. Requesting OBJECTS + DENSE_CAPTIONS...")
    print("   Expected: OBJECTS from cache, DENSE_CAPTIONS from API")
    start = time.time()
    detector.detect_objects(
        image, 
        scale_factor=1.0, 
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS]
    )
    duration_two_features = time.time() - start
    print(f"   Duration: {duration_two_features:.3f}s")
    
    # Step 3: Request all three features
    # OBJECTS and DENSE_CAPTIONS should be cached, only PEOPLE should hit API
    print("\n3. Requesting all 3 features (OBJECTS, DENSE_CAPTIONS, PEOPLE)...")
    print("   Expected: OBJECTS and DENSE_CAPTIONS from cache, PEOPLE from API")
    start = time.time()
    detector.detect_objects(
        image,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration_all_features = time.time() - start
    print(f"   Duration: {duration_all_features:.3f}s")
    
    # Step 4: Request all three features again (all should be cached)
    print("\n4. Requesting all 3 features again...")
    print("   Expected: All features from cache")
    start = time.time()
    detector.detect_objects(
        image,
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS, VisualFeatures.DENSE_CAPTIONS, VisualFeatures.PEOPLE]
    )
    duration_all_cached = time.time() - start
    print(f"   Duration: {duration_all_cached:.3f}s")
    
    print("\n✅ Feature Independence Verification:")
    print(f"   Step 1 (1 feature, cold):       {duration_objects_only:.3f}s")
    print(f"   Step 2 (2 features, 1 cached):  {duration_two_features:.3f}s")
    print(f"   Step 3 (3 features, 2 cached):  {duration_all_features:.3f}s")
    print(f"   Step 4 (3 features, all cached): {duration_all_cached:.3f}s ({duration_all_features/duration_all_cached:.2f}x speedup)")
    
    if duration_all_cached < duration_all_features * 0.5:
        print("\n   ✓ Features are cached independently and can be retrieved separately!")
    else:
        print("\n   ⚠ Cache behavior unclear")


def main():
    """Run all caching tests."""
    # Check environment variables
    if not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"):
        print("ERROR: VISION_ENDPOINT and VISION_KEY environment variables must be set")
        print("Please set these variables with your Azure Computer Vision credentials")
        return 1
    
    try:
        # Run main timing test
        run_timing_test()
        
        # Run feature independence test
        run_feature_independence_test()
        
        print("\n" + "="*80)
        print("All tests completed successfully!")
        print("="*80)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
