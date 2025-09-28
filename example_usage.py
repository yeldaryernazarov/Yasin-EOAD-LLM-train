"""
Example usage script for AD detection pipeline
Demonstrates how to use the pipeline for training, evaluation, and inference
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import the main pipeline class
from main import ADPipeline


def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    transcripts_dir = data_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    
    # Sample transcript texts (simplified examples)
    sample_transcripts = [
        "Hello, how are you today? I'm feeling quite well, thank you for asking.",
        "I went to the store yesterday and bought some groceries. It was a nice day.",
        "The weather is beautiful today. I think I'll go for a walk in the park.",
        "I'm having trouble remembering things lately. Sometimes I forget where I put my keys.",
        "What was I saying? Oh yes, I was talking about my doctor's appointment.",
        "I don't remember what I had for breakfast this morning. It's getting harder to recall things.",
        "I'm sorry, could you repeat that? I didn't quite catch what you said.",
        "I used to be able to remember everything, but now it's getting difficult.",
        "I'm worried about my memory. I keep forgetting important things.",
        "Sometimes I get confused about what day it is or where I am."
    ]
    
    # Create annotations CSV
    annotations_data = []
    for i, text in enumerate(sample_transcripts):
        # Create transcript file
        transcript_file = f"transcript_{i+1:03d}.txt"
        with open(transcripts_dir / transcript_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Determine label (simplified: last 5 are AD, first 5 are normal)
        label = 1 if i >= 5 else 0
        age = np.random.randint(65, 85)
        gender = np.random.choice(['M', 'F'])
        
        annotations_data.append({
            'id': i + 1,
            'age': age,
            'gender': gender,
            'label': label,
            'text_path': transcript_file
        })
    
    # Save annotations
    annotations_df = pd.DataFrame(annotations_data)
    annotations_df.to_csv(data_dir / "annotations.csv", index=False)
    
    print(f"Created {len(sample_transcripts)} sample transcripts")
    print(f"Class distribution: {annotations_df['label'].value_counts().to_dict()}")
    
    return str(data_dir / "annotations.csv"), str(transcripts_dir)


def demonstrate_training():
    """Demonstrate model training"""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL TRAINING")
    print("="*60)
    
    # Create sample data
    annotations_path, transcripts_dir = create_sample_data()
    
    # Initialize pipeline
    pipeline = ADPipeline()
    
    # Setup data
    print("\n1. Setting up data...")
    df = pipeline.setup_data(annotations_path, transcripts_dir)
    
    # Train model
    print("\n2. Training model...")
    model_path = pipeline.train_model(df, cv_folds=2)  # Reduced folds for demo
    
    return model_path, df


def demonstrate_evaluation(model_path, df):
    """Demonstrate model evaluation"""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL EVALUATION")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ADPipeline()
    
    # Evaluate model
    print("\n1. Evaluating model...")
    evaluation_results = pipeline.evaluate_model(
        model_path, 
        df, 
        demographic_columns=['age', 'gender']
    )
    
    # Print key metrics
    overall_perf = evaluation_results['overall_performance']
    print(f"\nEvaluation Results:")
    print(f"  Accuracy: {overall_perf['accuracy']:.3f}")
    print(f"  Precision: {overall_perf['precision']:.3f}")
    print(f"  Recall: {overall_perf['recall']:.3f}")
    print(f"  F1-Score: {overall_perf['f1']:.3f}")
    print(f"  AUC: {overall_perf['auc']:.3f}")
    
    return evaluation_results


def demonstrate_inference(model_path):
    """Demonstrate inference capabilities"""
    print("\n" + "="*60)
    print("DEMONSTRATING INFERENCE")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ADPipeline()
    
    # Setup inference
    print("\n1. Setting up inference system...")
    pipeline.setup_inference(model_path)
    
    # Test texts
    test_texts = [
        "Hello, how are you today? I'm feeling quite well, thank you for asking.",
        "I'm having trouble remembering things lately. Sometimes I forget where I put my keys.",
        "What was I saying? Oh yes, I was talking about my doctor's appointment.",
        "I went to the store yesterday and bought some groceries. It was a nice day.",
        "I don't remember what I had for breakfast this morning. It's getting harder to recall things."
    ]
    
    print("\n2. Making predictions...")
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        
        # Make prediction
        result = pipeline.predict(text, explain=True)
        
        print(f"  Prediction: {result['prediction']}")
        print(f"  AD Probability: {result['probability_ad']:.3f}")
        print(f"  Normal Probability: {result['probability_normal']:.3f}")
        print(f"  Confidence: {result['confidence']:.3f}")
        
        # Show top features if available
        if 'explanation' in result and result['explanation']:
            top_features = result['explanation'].get('top_contributing_tokens', [])[:3]
            if top_features:
                print(f"  Top features: {', '.join([f['token'] for f in top_features])}")


def demonstrate_explainability(model_path):
    """Demonstrate explainability features"""
    print("\n" + "="*60)
    print("DEMONSTRATING EXPLAINABILITY")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ADPipeline()
    
    # Setup explainer
    print("\n1. Setting up explainability system...")
    pipeline.setup_explainer(model_path)
    
    # Test text
    test_text = "I'm having trouble remembering things lately. Sometimes I forget where I put my keys."
    
    print(f"\n2. Explaining prediction for: {test_text}")
    
    # Get explanation
    explanation = pipeline.explain_prediction(test_text)
    
    print(f"\nPrediction: {explanation['prediction']}")
    print(f"AD Probability: {explanation['ad_probability']:.3f}")
    
    if explanation.get('top_features'):
        print(f"\nTop Contributing Features:")
        for i, feature in enumerate(explanation['top_features'][:10], 1):
            print(f"  {i:2d}. {feature['token']:15s} {feature['score']:8.3f} ({feature['contribution']})")
    
    if 'interpretation' in explanation:
        print(f"\nInterpretation: {explanation['interpretation']}")


def demonstrate_batch_processing(model_path):
    """Demonstrate batch processing"""
    print("\n" + "="*60)
    print("DEMONSTRATING BATCH PROCESSING")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ADPipeline()
    pipeline.setup_inference(model_path)
    
    # Create test data
    test_data = pd.DataFrame({
        'id': range(1, 6),
        'text': [
            "Hello, how are you today? I'm feeling quite well.",
            "I'm having trouble remembering things lately.",
            "What was I saying? Oh yes, I was talking about my appointment.",
            "I went to the store yesterday and bought groceries.",
            "I don't remember what I had for breakfast this morning."
        ],
        'label': [0, 1, 1, 0, 1]  # Ground truth labels
    })
    
    print("\n1. Processing batch of texts...")
    
    # Process batch
    results = []
    for idx, row in test_data.iterrows():
        result = pipeline.predict(row['text'])
        results.append({
            'id': row['id'],
            'text': row['text'][:50] + "...",
            'true_label': 'AD' if row['label'] == 1 else 'Normal',
            'predicted_label': result['prediction'],
            'ad_probability': result['probability_ad'],
            'correct': (row['label'] == 1 and result['prediction'] == 'AD') or 
                      (row['label'] == 0 and result['prediction'] == 'Normal')
        })
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nBatch Processing Results:")
    print(results_df.to_string(index=False))
    
    # Calculate accuracy
    accuracy = results_df['correct'].mean()
    print(f"\nBatch Accuracy: {accuracy:.3f}")


def main():
    """Main demonstration function"""
    print("Alzheimer's Disease Detection Pipeline - Example Usage")
    print("=" * 60)
    
    try:
        # Step 1: Training
        model_path, df = demonstrate_training()
        
        # Step 2: Evaluation
        evaluation_results = demonstrate_evaluation(model_path, df)
        
        # Step 3: Inference
        demonstrate_inference(model_path)
        
        # Step 4: Explainability
        demonstrate_explainability(model_path)
        
        # Step 5: Batch Processing
        demonstrate_batch_processing(model_path)
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model saved at: {model_path}")
        print("Check the 'results/' directory for evaluation plots and reports.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()