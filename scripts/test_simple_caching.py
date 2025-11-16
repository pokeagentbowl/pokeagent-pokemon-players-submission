"""
Test script comparing different caching approaches for Azure Computer Vision.

This demonstrates:
1. SimpleCachedObjectDetector - Uses SQLite cache, caches entire response
2. DecoratorCachedObjectDetector - Uses lru_cache decorator, minimal code
3. Original CachedObjectDetector - Per-feature caching (more complex)
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_utils.simple_cached_detector import SimpleCachedObjectDetector, DecoratorCachedObjectDetector
from custom_utils.cached_object_detector import CachedObjectDetector


def create_test_image(size=(240, 160), color=(255, 0, 0)):
    """Create a test image."""
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)


def test_simple_cached_detector():
    """Test SimpleCachedObjectDetector - minimal code, SQLite cache."""
    print("="*80)
    print("TEST 1: SimpleCachedObjectDetector (SQLite cache, entire response)")
    print("="*80)
    
    detector = SimpleCachedObjectDetector()
    image = create_test_image(color=(255, 0, 0))
    
    # First call
    print("\n1. First call (cold start)...")
    start = time.time()
    result1 = detector.detect_objects(image, scale_factor=1.0)
    duration1 = time.time() - start
    print(f"   Detected {len(result1)} objects in {duration1:.3f}s")
    
    # Second call (should be cached)
    print("\n2. Second call (should be cached)...")
    start = time.time()
    result2 = detector.detect_objects(image, scale_factor=1.0)
    duration2 = time.time() - start
    print(f"   Detected {len(result2)} objects in {duration2:.3f}s")
    print(f"   Speedup: {duration1/duration2:.2f}x")
    
    # Different image (should be cold)
    image2 = create_test_image(color=(0, 255, 0))
    print("\n3. Different image (cold start)...")
    start = time.time()
    result3 = detector.detect_objects(image2, scale_factor=1.0)
    duration3 = time.time() - start
    print(f"   Detected {len(result3)} objects in {duration3:.3f}s")
    
    # Same different image (should be cached)
    print("\n4. Same different image (cached)...")
    start = time.time()
    result4 = detector.detect_objects(image2, scale_factor=1.0)
    duration4 = time.time() - start
    print(f"   Detected {len(result4)} objects in {duration4:.3f}s")
    print(f"   Speedup: {duration3/duration4:.2f}x")
    
    print(f"\n✅ SimpleCachedObjectDetector Summary:")
    print(f"   - Minimal code (just cache_get/cache_set)")
    print(f"   - SQLite persistent cache")
    print(f"   - Average speedup: {(duration1/duration2 + duration3/duration4)/2:.2f}x")


def test_decorator_cached_detector():
    """Test DecoratorCachedObjectDetector - lru_cache decorator."""
    print("\n" + "="*80)
    print("TEST 2: DecoratorCachedObjectDetector (lru_cache decorator)")
    print("="*80)
    
    detector = DecoratorCachedObjectDetector(cache_size=128)
    image = create_test_image(color=(255, 0, 0))
    
    # First call
    print("\n1. First call (cold start)...")
    start = time.time()
    result1 = detector.detect_objects(image, scale_factor=1.0)
    duration1 = time.time() - start
    print(f"   Detected {len(result1)} objects in {duration1:.3f}s")
    
    # Second call (should be cached)
    print("\n2. Second call (should be cached)...")
    start = time.time()
    result2 = detector.detect_objects(image, scale_factor=1.0)
    duration2 = time.time() - start
    print(f"   Detected {len(result2)} objects in {duration2:.3f}s")
    print(f"   Speedup: {duration1/duration2:.2f}x")
    
    print(f"\n✅ DecoratorCachedObjectDetector Summary:")
    print(f"   - Most minimal code (just @lru_cache)")
    print(f"   - In-memory cache only")
    print(f"   - Speedup: {duration1/duration2:.2f}x")


def test_original_cached_detector():
    """Test original CachedObjectDetector - per-feature caching."""
    print("\n" + "="*80)
    print("TEST 3: CachedObjectDetector (per-feature, complex)")
    print("="*80)
    
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    
    detector = CachedObjectDetector()
    image = create_test_image(color=(255, 0, 0))
    
    # First call with all features
    print("\n1. First call with all features...")
    start = time.time()
    result1 = detector.detect_objects(image, scale_factor=1.0)
    duration1 = time.time() - start
    print(f"   Detected {len(result1)} objects in {duration1:.3f}s")
    
    # Second call with all features (should be cached)
    print("\n2. Second call with all features (cached)...")
    start = time.time()
    result2 = detector.detect_objects(image, scale_factor=1.0)
    duration2 = time.time() - start
    print(f"   Detected {len(result2)} objects in {duration2:.3f}s")
    print(f"   Speedup: {duration1/duration2:.2f}x")
    
    # Third call with partial features (should be mostly cached)
    print("\n3. Call with only OBJECTS (all cached)...")
    start = time.time()
    result3 = detector.detect_objects(
        image, 
        scale_factor=1.0,
        visual_features=[VisualFeatures.OBJECTS]
    )
    duration3 = time.time() - start
    print(f"   Detected {len(result3)} objects in {duration3:.3f}s")
    
    print(f"\n✅ CachedObjectDetector Summary:")
    print(f"   - Most complex code (per-feature handling)")
    print(f"   - SQLite persistent cache")
    print(f"   - Per-feature granularity")
    print(f"   - Average speedup: {duration1/duration2:.2f}x")


def comparison_summary():
    """Print comparison of approaches."""
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\n1. SimpleCachedObjectDetector")
    print("   Pros:")
    print("   - Minimal code (~50 lines vs ~400 for original)")
    print("   - SQLite persistent cache")
    print("   - Easy to understand")
    print("   Cons:")
    print("   - Caches entire response (no per-feature granularity)")
    print("   - Still requires manual cache_get/cache_set")
    
    print("\n2. DecoratorCachedObjectDetector")
    print("   Pros:")
    print("   - Most minimal code (just @lru_cache decorator)")
    print("   - Pythonic and elegant")
    print("   - No manual cache management")
    print("   Cons:")
    print("   - In-memory only (lost on restart)")
    print("   - Limited cache size")
    print("   - Workaround needed for numpy arrays (not hashable)")
    
    print("\n3. CachedObjectDetector (Original)")
    print("   Pros:")
    print("   - Per-feature caching (most flexible)")
    print("   - SQLite persistent cache")
    print("   - Handles partial cache hits")
    print("   Cons:")
    print("   - Most complex (~400 lines)")
    print("   - Manual hash checking and cache management")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("LangChain's set_llm_cache ONLY works for LLMs, not for custom tools/runnables.")
    print("For custom functions like Azure CV API:")
    print("- Use @lru_cache for simplest in-memory caching")
    print("- Use SimpleCachedObjectDetector for persistent SQLite cache with minimal code")
    print("- Use CachedObjectDetector for maximum flexibility (per-feature caching)")
    print("="*80)


def main():
    """Run all tests."""
    if not os.environ.get("VISION_ENDPOINT") or not os.environ.get("VISION_KEY"):
        print("ERROR: VISION_ENDPOINT and VISION_KEY environment variables must be set")
        return 1
    
    try:
        test_simple_cached_detector()
        test_decorator_cached_detector()
        test_original_cached_detector()
        comparison_summary()
        
        print("\n✅ All tests completed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
