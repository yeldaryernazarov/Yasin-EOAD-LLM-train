"""
Test that all main modules can be imported without errors
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_main():
    """Test that main module can be imported"""
    try:
        import main
        assert hasattr(main, 'ADPipeline')
        assert hasattr(main, 'main')
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {e}")

def test_import_config():
    """Test that config module can be imported"""
    try:
        import config
        assert hasattr(config, 'MODEL_CONFIG')
        assert hasattr(config, 'DATA_CONFIG')
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")

def test_import_data_preprocessing():
    """Test that data preprocessing module can be imported"""
    try:
        import data_preprocessing
        assert hasattr(data_preprocessing, 'TranscriptPreprocessor')
    except ImportError as e:
        pytest.fail(f"Failed to import data_preprocessing module: {e}")

def test_import_train_model():
    """Test that train model module can be imported"""
    try:
        import train_model
        assert hasattr(train_model, 'ADTrainer')
    except ImportError as e:
        pytest.fail(f"Failed to import train_model module: {e}")

def test_import_evaluate_model():
    """Test that evaluate model module can be imported"""
    try:
        import evaluate_model
        assert hasattr(evaluate_model, 'ADModelEvaluator')
    except ImportError as e:
        pytest.fail(f"Failed to import evaluate_model module: {e}")

def test_import_inference():
    """Test that inference module can be imported"""
    try:
        import inference
        assert hasattr(inference, 'ADInference')
    except ImportError as e:
        pytest.fail(f"Failed to import inference module: {e}")

def test_import_explainability():
    """Test that explainability module can be imported"""
    try:
        import explainability
        assert hasattr(explainability, 'ADModelExplainer')
    except ImportError as e:
        pytest.fail(f"Failed to import explainability module: {e}")

def test_pipeline_initialization():
    """Test that ADPipeline can be initialized"""
    try:
        from main import ADPipeline
        pipeline = ADPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'setup_data')
        assert hasattr(pipeline, 'train_model')
        assert hasattr(pipeline, 'evaluate_model')
    except Exception as e:
        pytest.fail(f"Failed to initialize ADPipeline: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
