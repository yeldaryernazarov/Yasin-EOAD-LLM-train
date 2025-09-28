"""
Test script for small datasets
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_small_test_data():
    """Create a small test dataset"""
    print("Creating small test dataset...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    transcripts_dir = data_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    
    # Very small dataset (only 4 samples)
    sample_data = [
        {"id": 1, "age": 75, "gender": "M", "label": 0, "text": "Hello, how are you today? I'm feeling quite well."},
        {"id": 2, "age": 68, "gender": "F", "label": 0, "text": "I went to the store yesterday and bought some groceries."},
        {"id": 3, "age": 72, "gender": "M", "label": 1, "text": "I'm having trouble remembering things lately."},
        {"id": 4, "age": 80, "gender": "F", "label": 1, "text": "What was I saying? Oh yes, I was talking about my appointment."},
    ]
    
    # Create annotations CSV
    annotations_data = []
    for item in sample_data:
        # Create transcript file
        transcript_file = f"transcript_{item['id']:03d}.txt"
        with open(transcripts_dir / transcript_file, 'w', encoding='utf-8') as f:
            f.write(item['text'])
        
        annotations_data.append({
            'id': item['id'],
            'age': item['age'],
            'gender': item['gender'],
            'label': item['label'],
            'text_path': transcript_file
        })
    
    # Save annotations
    annotations_df = pd.DataFrame(annotations_data)
    annotations_df.to_csv(data_dir / "annotations.csv", index=False)
    
    print(f"Created {len(sample_data)} sample transcripts")
    print(f"Class distribution: {annotations_df['label'].value_counts().to_dict()}")
    
    return str(data_dir / "annotations.csv"), str(transcripts_dir)

def test_small_dataset():
    """Test the pipeline with a small dataset"""
    print("=" * 60)
    print("TESTING WITH SMALL DATASET")
    print("=" * 60)
    
    try:
        # Create small test data
        annotations_path, transcripts_dir = create_small_test_data()
        
        # Import and test
        from main import ADPipeline
        
        # Initialize pipeline
        pipeline = ADPipeline()
        
        # Setup data
        print("\n1. Setting up data...")
        df = pipeline.setup_data(annotations_path, transcripts_dir)
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        # Train model (this should now work with small dataset)
        print("\n2. Training model...")
        model_path = pipeline.train_model(df, cv_folds=2)
        print(f"Model trained successfully: {model_path}")
        
        # Test inference
        print("\n3. Testing inference...")
        pipeline.setup_inference(model_path)
        
        test_text = "I'm having trouble remembering things lately."
        result = pipeline.predict(test_text)
        
        print(f"Test text: {test_text}")
        print(f"Prediction: {result['prediction']}")
        print(f"AD Probability: {result['probability_ad']:.3f}")
        
        print("\n✅ Small dataset test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Small dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_dataset()
    if success:
        print("\n🎉 All tests passed! The pipeline now handles small datasets correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")