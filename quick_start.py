"""
Quick start script for AD detection pipeline
Shows the simplest way to get started
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ADPipeline


def quick_demo():
    """Quick demonstration of the pipeline"""
    print("🚀 AD Detection Pipeline - Quick Start")
    print("=" * 50)
    
    # Check if sample data exists
    if not os.path.exists("data/annotations.csv"):
        print("❌ Sample data not found!")
        print("Please run: python setup.py")
        return
    
    print("✅ Sample data found")
    
    # Initialize pipeline
    print("\n📊 Initializing pipeline...")
    pipeline = ADPipeline()
    
    # Run full pipeline
    print("\n🔄 Running full pipeline...")
    try:
        results = pipeline.run_full_pipeline(
            annotations_path="data/annotations.csv",
            transcripts_dir="data/transcripts/",
            cv_folds=2  # Reduced for quick demo
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📁 Model saved at: {results['model_path']}")
        print("📈 Check the 'results/' directory for evaluation plots")
        
        # Test inference
        print("\n🧪 Testing inference...")
        test_text = "I'm having trouble remembering things lately."
        result = pipeline.predict(test_text, explain=True)
        
        print(f"📝 Test text: {test_text}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 AD Probability: {result['probability_ad']:.3f}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        
        if 'explanation' in result and result['explanation']:
            top_features = result['explanation'].get('top_contributing_tokens', [])[:3]
            if top_features:
                print(f"🔍 Top features: {', '.join([f['token'] for f in top_features])}")
        
        print("\n🎉 Quick start completed successfully!")
        print("\nNext steps:")
        print("1. Replace data/annotations.csv with your real data")
        print("2. Replace data/transcripts/*.txt with your transcript files")
        print("3. Run: python main.py pipeline --annotations your_data.csv --transcripts_dir your_transcripts/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    quick_demo()