"""
Test script to verify the fixes work
"""
import sys
import os

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import MODEL_CONFIG, DATA_CONFIG, EVAL_CONFIG
        print("✅ Config imports: OK")
    except Exception as e:
        print(f"❌ Config imports: {e}")
        return False
    
    try:
        from data_preprocessing import TranscriptPreprocessor
        print("✅ Data preprocessing: OK")
    except Exception as e:
        print(f"❌ Data preprocessing: {e}")
        return False
    
    try:
        from train_model import ADTrainer
        print("✅ Train model: OK")
    except Exception as e:
        print(f"❌ Train model: {e}")
        return False
    
    try:
        from evaluate_model import ADModelEvaluator
        print("✅ Evaluate model: OK")
    except Exception as e:
        print(f"❌ Evaluate model: {e}")
        return False
    
    try:
        from inference import ADInference
        print("✅ Inference: OK")
    except Exception as e:
        print(f"❌ Inference: {e}")
        return False
    
    try:
        from explainability import ADModelExplainer
        print("✅ Explainability: OK")
    except Exception as e:
        print(f"❌ Explainability: {e}")
        return False
    
    try:
        from main import ADPipeline
        print("✅ Main pipeline: OK")
    except Exception as e:
        print(f"❌ Main pipeline: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from data_preprocessing import TranscriptPreprocessor
        from config import DATA_CONFIG, FEATURE_CONFIG
        
        # Test preprocessor initialization
        config = {**DATA_CONFIG, **FEATURE_CONFIG}
        preprocessor = TranscriptPreprocessor(config)
        print("✅ Preprocessor initialization: OK")
        
        # Test text cleaning
        test_text = "Hello, how are you today? I'm feeling quite well."
        cleaned = preprocessor.clean_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test feature extraction
        features = preprocessor.extract_linguistic_features(test_text)
        print(f"✅ Linguistic features: {len(features)} features extracted")
        
        paralinguistic_features = preprocessor.extract_paralinguistic_features(test_text)
        print(f"✅ Paralinguistic features: {len(paralinguistic_features)} features extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("TESTING AD DETECTION PIPELINE FIXES")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)
    print("\nThe pipeline should now work correctly.")
    print("You can run:")
    print("  python setup.py")
    print("  python quick_start.py")
    print("  python main.py pipeline --annotations data/annotations.csv --transcripts_dir data/transcripts/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)