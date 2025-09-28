"""
Test script to identify import errors
"""
import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: {e}")
        traceback.print_exc()
        return False

def main():
    print("Testing imports...")
    print("=" * 50)
    
    # Test basic dependencies
    test_import("pandas", "pandas")
    test_import("numpy", "numpy")
    test_import("torch", "torch")
    test_import("transformers", "transformers")
    test_import("sklearn", "scikit-learn")
    test_import("nltk", "nltk")
    test_import("textstat", "textstat")
    test_import("shap", "shap")
    test_import("matplotlib", "matplotlib")
    test_import("seaborn", "seaborn")
    
    print("\nTesting project modules...")
    print("=" * 50)
    
    # Test project modules
    test_import("config", "config")
    test_import("data_preprocessing", "data_preprocessing")
    test_import("train_model", "train_model")
    test_import("evaluate_model", "evaluate_model")
    test_import("inference", "inference")
    test_import("explainability", "explainability")
    test_import("main", "main")

if __name__ == "__main__":
    main()