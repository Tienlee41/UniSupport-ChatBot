"""Test script to check if OCR image functionality exists in data_retriever"""
import sys
import os

print("=" * 80)
print("OCR IMAGE CHECK")
print("=" * 80)

# 1. Check data_retriever location
try:
    import data_retriever
    print(f"\n[1] data_retriever location: {data_retriever.__file__}")
except Exception as e:
    print(f"\n[1] ERROR importing data_retriever: {e}")
    sys.exit(1)

# 2. Check ImageOCR module exists
print("\n[2] Checking ImageOCR module...")
try:
    from data_retriever.extractor.image_ocr import ImageOCR
    print("   ✓ ImageOCR class imported successfully")
    print(f"   Location: {ImageOCR.__file__ if hasattr(ImageOCR, '__file__') else 'N/A'}")
except ImportError as e:
    print(f"   ✗ ImageOCR NOT FOUND: {e}")
    print("   → Code OCR chưa được download hoặc chưa có trong data_retriever")
except Exception as e:
    print(f"   ✗ ERROR importing ImageOCR: {e}")

# 3. Check ImageFinder module exists
print("\n[3] Checking ImageFinder module...")
try:
    from data_retriever.extractor.image_finder import ImageFinder
    print("   ✓ ImageFinder class imported successfully")
except ImportError as e:
    print(f"   ✗ ImageFinder NOT FOUND: {e}")
except Exception as e:
    print(f"   ✗ ERROR importing ImageFinder: {e}")

# 4. Check ImageOCR initialization (needs API key)
print("\n[4] Testing ImageOCR initialization...")
try:
    from data_retriever.extractor.image_ocr import ImageOCR
    ocr = ImageOCR()
    print(f"   ✓ ImageOCR instance created")
    print(f"   Enabled: {ocr.enabled}")
    
    # Check API key
    api_key = os.getenv("VISION_AGENT_API_KEY") or os.getenv("LANDINGAI_API_KEY")
    if api_key:
        print(f"   ✓ API key found: {api_key[:20]}...")
    else:
        print(f"   ✗ API key NOT FOUND (VISION_AGENT_API_KEY or LANDINGAI_API_KEY)")
        print("   → OCR sẽ không hoạt động nếu không có API key")
        
    # Check landingai_ade package
    try:
        from landingai_ade import LandingAIADE
        print(f"   ✓ landingai_ade package installed")
    except ImportError:
        print(f"   ✗ landingai_ade package NOT installed")
        print("   → Cần: pip install landingai-ade")
        
except Exception as e:
    print(f"   ✗ ERROR creating ImageOCR: {e}")
    import traceback
    traceback.print_exc()

# 5. Check ContentExtractor uses ImageOCR
print("\n[5] Checking ContentExtractor integration...")
try:
    from data_retriever.extractor.content_extractor import ContentExtractor
    import inspect
    
    # Check if ContentExtractor has image_ocr attribute
    source = inspect.getsource(ContentExtractor.__init__)
    if "image_ocr" in source or "ImageOCR" in source:
        print("   ✓ ContentExtractor.__init__ references ImageOCR")
    else:
        print("   ✗ ContentExtractor.__init__ does NOT reference ImageOCR")
        print("   → Code OCR chưa được tích hợp vào ContentExtractor")
        
    # Check if _job method has include_image parameter
    if hasattr(ContentExtractor, '_job'):
        job_source = inspect.getsource(ContentExtractor._job)
        if "include_image" in job_source:
            print("   ✓ ContentExtractor._job has include_image parameter")
        else:
            print("   ✗ ContentExtractor._job does NOT have include_image parameter")
            
except Exception as e:
    print(f"   ✗ ERROR checking ContentExtractor: {e}")
    import traceback
    traceback.print_exc()

# 6. Check ImageFinder integration
print("\n[6] Checking ImageFinder integration...")
try:
    from data_retriever.extractor.content_extractor import ContentExtractor
    import inspect
    
    source = inspect.getsource(ContentExtractor.__init__)
    if "image_finder" in source or "ImageFinder" in source:
        print("   ✓ ContentExtractor.__init__ references ImageFinder")
    else:
        print("   ✗ ContentExtractor.__init__ does NOT reference ImageFinder")
        
except Exception as e:
    print(f"   ✗ ERROR checking ImageFinder integration: {e}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print("Nếu thấy ✓ ở tất cả các bước → OCR code đã được download và sẵn sàng")
print("Nếu thấy ✗ ở bước 2-3 → Code OCR chưa được download từ server")
print("Nếu thấy ✗ ở bước 4 → Cần set API key hoặc install landingai-ade")
print("Nếu thấy ✗ ở bước 5-6 → Code OCR chưa được tích hợp vào pipeline")
print("=" * 80)

