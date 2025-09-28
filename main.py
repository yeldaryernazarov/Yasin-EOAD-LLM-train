"""
Main script for AD detection pipeline
Provides a unified interface for training, evaluation, and inference
"""
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_CONFIG, DATA_CONFIG, EVAL_CONFIG, FEATURE_CONFIG, 
    INFERENCE_CONFIG, MODELS_DIR, RESULTS_DIR, LOGS_DIR
)
from data_preprocessing import TranscriptPreprocessor
from train_model import ADTrainer
from evaluate_model import ADModelEvaluator
from inference import ADInference
from explainability import ADModelExplainer


class ADPipeline:
    """Main pipeline class for AD detection"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the AD detection pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        self.inference = None
        self.explainer = None
    
    def setup_data(self, annotations_path: str, transcripts_dir: str) -> pd.DataFrame:
        """
        Setup and preprocess data
        
        Args:
            annotations_path: Path to annotations CSV
            transcripts_dir: Directory containing transcript files
            
        Returns:
            Preprocessed DataFrame
        """
        print("Setting up data...")
        
        # Initialize preprocessor
        config = {**DATA_CONFIG, **FEATURE_CONFIG}
        self.preprocessor = TranscriptPreprocessor(config)
        
        # Load and preprocess data
        df = self.preprocessor.load_data(annotations_path, transcripts_dir)
        df_processed = self.preprocessor.preprocess_dataset(df)
        
        print(f"Data loaded and preprocessed: {df_processed.shape}")
        print(f"Class distribution: {df_processed[DATA_CONFIG['label_column']].value_counts()}")
        
        return df_processed
    
    def train_model(self, df: pd.DataFrame, cv_folds: int = 5) -> str:
        """
        Train the AD detection model
        
        Args:
            df: Preprocessed dataset
            cv_folds: Number of cross-validation folds
            
        Returns:
            Path to best model
        """
        print("Training model...")
        
        # Initialize trainer
        config = {**MODEL_CONFIG, **DATA_CONFIG, **EVAL_CONFIG}
        config['cv_folds'] = cv_folds
        self.trainer = ADTrainer(config)
        self.trainer.setup_model()
        
        # Perform cross-validation
        results = self.trainer.cross_validate(df)
        
        # Save results
        self.trainer.save_results(results)
        self.trainer.plot_training_curves()
        
        print(f"Training completed. Best model: {self.trainer.best_model_path}")
        return self.trainer.best_model_path
    
    def evaluate_model(self, model_path: str, df: pd.DataFrame, 
                      demographic_columns: List[str] = None) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            model_path: Path to trained model
            df: Evaluation dataset
            demographic_columns: Columns for demographic analysis
            
        Returns:
            Evaluation results
        """
        print("Evaluating model...")
        
        # Initialize evaluator
        self.evaluator = ADModelEvaluator(model_path, EVAL_CONFIG)
        
        # Generate comprehensive evaluation report
        report = self.evaluator.generate_evaluation_report(
            df, 
            DATA_CONFIG['text_column'], 
            DATA_CONFIG['label_column'],
            demographic_columns or ['age', 'gender']
        )
        
        print("Evaluation completed.")
        return report
    
    def setup_inference(self, model_path: str, calibration_path: str = None):
        """
        Setup inference system
        
        Args:
            model_path: Path to trained model
            calibration_path: Path to calibration model
        """
        print("Setting up inference system...")
        
        self.inference = ADInference(
            model_path=model_path,
            calibration_path=calibration_path,
            config=INFERENCE_CONFIG
        )
        
        print("Inference system ready.")
    
    def setup_explainer(self, model_path: str):
        """
        Setup explainability system
        
        Args:
            model_path: Path to trained model
        """
        print("Setting up explainability system...")
        
        self.explainer = ADModelExplainer(model_path)
        
        print("Explainability system ready.")
    
    def predict(self, text: str, explain: bool = False) -> Dict:
        """
        Make prediction on text
        
        Args:
            text: Input text
            explain: Whether to include explanation
            
        Returns:
            Prediction results
        """
        if self.inference is None:
            raise ValueError("Inference system not initialized. Call setup_inference() first.")
        
        return self.inference.predict(text, return_explanation=explain)
    
    def explain_prediction(self, text: str) -> Dict:
        """
        Explain a prediction
        
        Args:
            text: Input text
            
        Returns:
            Explanation results
        """
        if self.explainer is None:
            raise ValueError("Explainability system not initialized. Call setup_explainer() first.")
        
        return self.explainer.explain_prediction_with_context(text)
    
    def run_full_pipeline(self, annotations_path: str, transcripts_dir: str,
                         cv_folds: int = 5, demographic_columns: List[str] = None) -> Dict:
        """
        Run the complete pipeline from data loading to model deployment
        
        Args:
            annotations_path: Path to annotations CSV
            transcripts_dir: Directory containing transcript files
            cv_folds: Number of cross-validation folds
            demographic_columns: Columns for demographic analysis
            
        Returns:
            Pipeline results
        """
        print("="*60)
        print("RUNNING FULL AD DETECTION PIPELINE")
        print("="*60)
        
        # Step 1: Setup data
        df = self.setup_data(annotations_path, transcripts_dir)
        
        # Step 2: Train model
        model_path = self.train_model(df, cv_folds)
        
        # Step 3: Evaluate model
        evaluation_results = self.evaluate_model(model_path, df, demographic_columns)
        
        # Step 4: Setup inference and explainability
        self.setup_inference(model_path)
        self.setup_explainer(model_path)
        
        # Step 5: Test inference
        print("\nTesting inference on sample texts...")
        sample_texts = df[DATA_CONFIG['text_column']].head(3).tolist()
        for i, text in enumerate(sample_texts):
            result = self.predict(text[:100] + "...", explain=True)
            print(f"\nSample {i+1}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  AD Probability: {result['probability_ad']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return {
            'model_path': model_path,
            'evaluation_results': evaluation_results,
            'pipeline_status': 'completed'
        }


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AD Detection Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument('--annotations', type=str, required=True,
                               help='Path to annotations CSV file')
    pipeline_parser.add_argument('--transcripts_dir', type=str, required=True,
                               help='Directory containing transcript files')
    pipeline_parser.add_argument('--cv_folds', type=int, default=3,
                               help='Number of cross-validation folds (will be adjusted based on dataset size)')
    pipeline_parser.add_argument('--demographic_columns', nargs='+', 
                               default=['age', 'gender'],
                               help='Demographic columns for analysis')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model only')
    train_parser.add_argument('--annotations', type=str, required=True,
                            help='Path to annotations CSV file')
    train_parser.add_argument('--transcripts_dir', type=str, required=True,
                            help='Directory containing transcript files')
    train_parser.add_argument('--cv_folds', type=int, default=3,
                            help='Number of cross-validation folds (will be adjusted based on dataset size)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model only')
    eval_parser.add_argument('--model_path', type=str, required=True,
                           help='Path to trained model')
    eval_parser.add_argument('--data_file', type=str, required=True,
                           help='Path to evaluation data CSV')
    eval_parser.add_argument('--text_column', type=str, default='text',
                           help='Name of text column')
    eval_parser.add_argument('--label_column', type=str, default='label',
                           help='Name of label column')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Make predictions')
    infer_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained model')
    infer_parser.add_argument('--text', type=str, default=None,
                            help='Text to analyze')
    infer_parser.add_argument('--input_file', type=str, default=None,
                            help='File containing texts to analyze')
    infer_parser.add_argument('--output_file', type=str, default=None,
                            help='Output file for results')
    infer_parser.add_argument('--explain', action='store_true',
                            help='Generate explanations')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain predictions')
    explain_parser.add_argument('--model_path', type=str, required=True,
                              help='Path to trained model')
    explain_parser.add_argument('--text', type=str, default=None,
                              help='Text to explain')
    explain_parser.add_argument('--data_file', type=str, default=None,
                              help='Dataset file for batch analysis')
    explain_parser.add_argument('--sample_size', type=int, default=100,
                              help='Number of samples for analysis')
    
    args = parser.parse_args()
    
    if args.command == 'pipeline':
        # Run full pipeline
        pipeline = ADPipeline()
        results = pipeline.run_full_pipeline(
            args.annotations,
            args.transcripts_dir,
            args.cv_folds,
            args.demographic_columns
        )
        print(f"Pipeline completed. Model saved at: {results['model_path']}")
    
    elif args.command == 'train':
        # Train model only
        pipeline = ADPipeline()
        df = pipeline.setup_data(args.annotations, args.transcripts_dir)
        model_path = pipeline.train_model(df, args.cv_folds)
        print(f"Training completed. Model saved at: {model_path}")
    
    elif args.command == 'evaluate':
        # Evaluate model only
        evaluator = ADModelEvaluator(args.model_path)
        df = pd.read_csv(args.data_file)
        report = evaluator.generate_evaluation_report(
            df, args.text_column, args.label_column
        )
        print("Evaluation completed.")
    
    elif args.command == 'infer':
        # Make predictions
        inference = ADInference(args.model_path)
        
        if args.text:
            # Single text prediction
            result = inference.predict(args.text, return_explanation=args.explain)
            print(f"Prediction: {result['prediction']}")
            print(f"AD Probability: {result['probability_ad']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    import json
                    json.dump(result, f, indent=2, default=str)
        
        elif args.input_file:
            # Batch prediction
            df = pd.read_csv(args.input_file)
            results = inference.batch_predict(df[args.text_column].tolist())
            
            if args.output_file:
                inference.save_prediction_report(results, args.output_file)
            else:
                print(f"Processed {len(results)} texts")
    
    elif args.command == 'explain':
        # Explain predictions
        explainer = ADModelExplainer(args.model_path)
        
        if args.text:
            # Single text explanation
            explanation = explainer.explain_prediction_with_context(args.text)
            print(f"Text: {args.text[:100]}...")
            print(f"Prediction: {explanation['prediction']}")
            print(f"AD Probability: {explanation['ad_probability']:.3f}")
            
            if explanation.get('top_features'):
                print("\nTop Contributing Features:")
                for i, feature in enumerate(explanation['top_features'][:10], 1):
                    print(f"  {i:2d}. {feature['token']:15s} {feature['score']:8.3f}")
        
        elif args.data_file:
            # Batch analysis
            df = pd.read_csv(args.data_file)
            report = explainer.generate_explanation_report(
                df, sample_size=args.sample_size
            )
            print("Explanation analysis completed.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()