"""
Inference script for AD detection model
Provides real-time predictions with explainability
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import INFERENCE_CONFIG, FEATURE_CONFIG
from data_preprocessing import TranscriptPreprocessor


class ADInference:
    """Inference class for AD detection"""
    
    def __init__(self, model_path: str, calibration_path: str = None, config: Dict = None):
        """
        Initialize inference model
        
        Args:
            model_path: Path to trained model
            calibration_path: Path to calibration model
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.calibration_path = calibration_path
        self.config = config or INFERENCE_CONFIG
        
        # Load model and tokenizer
        self._load_model()
        
        # Load calibration model if available
        self.calibration_model = None
        if calibration_path and os.path.exists(calibration_path):
            self._load_calibration_model()
        
        # Initialize preprocessor for feature extraction
        self.preprocessor = TranscriptPreprocessor(FEATURE_CONFIG)
        
        # Initialize explainer
        self.explainer = None
        self._setup_explainer()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def _load_calibration_model(self):
        """Load calibration model"""
        try:
            with open(self.calibration_path, 'rb') as f:
                self.calibration_model = pickle.load(f)
            print("Calibration model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load calibration model: {e}")
            self.calibration_model = None
    
    def _setup_explainer(self):
        """Setup SHAP explainer for model interpretability"""
        try:
            # For now, we'll use a simplified approach
            # SHAP integration with transformers requires more complex setup
            self.explainer = None
            print("SHAP explainer setup deferred (using simplified explanations)")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        return self.preprocessor.clean_text(text)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and paralinguistic features"""
        features = {}
        
        if self.config.get('linguistic_features', True):
            linguistic_features = self.preprocessor.extract_linguistic_features(text)
            features.update(linguistic_features)
        
        if self.config.get('paralinguistic_features', True):
            paralinguistic_features = self.preprocessor.extract_paralinguistic_features(text)
            features.update(paralinguistic_features)
        
        return features
    
    def predict(self, text: str, return_features: bool = False, 
                return_explanation: bool = False) -> Dict:
        """
        Predict AD probability for given text
        
        Args:
            text: Input text to analyze
            return_features: Whether to return extracted features
            return_explanation: Whether to return SHAP explanation
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text or cleaned_text.strip() == "":
            return {
                'prediction': 'error',
                'probability_ad': 0.0,
                'probability_normal': 1.0,
                'confidence': 0.0,
                'error': 'Empty or invalid text input'
            }
        
        # Tokenize and prepare input
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()[0]
        logits_np = logits.cpu().numpy()[0]
        
        # Apply calibration if available
        if self.calibration_model is not None:
            try:
                # Use the raw probability for class 1 (AD)
                raw_prob = probs_np[1]
                calibrated_prob = self.calibration_model.predict_proba([[raw_prob]])[0][1]
                probs_np = np.array([1 - calibrated_prob, calibrated_prob])
            except Exception as e:
                print(f"Warning: Calibration failed: {e}")
        
        # Determine prediction
        prediction = 'AD' if probs_np[1] > self.config.get('confidence_threshold', 0.5) else 'Normal'
        confidence = max(probs_np)
        
        # Prepare result
        result = {
            'prediction': prediction,
            'probability_ad': float(probs_np[1]),
            'probability_normal': float(probs_np[0]),
            'confidence': float(confidence),
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
        
        # Add features if requested
        if return_features:
            features = self.extract_features(cleaned_text)
            result['features'] = features
        
        # Add explanation if requested and available
        if return_explanation and self.explainer is not None:
            try:
                explanation = self._get_explanation(cleaned_text)
                result['explanation'] = explanation
            except Exception as e:
                print(f"Warning: Could not generate explanation: {e}")
                result['explanation'] = None
        
        return result
    
    def _get_explanation(self, text: str, top_k: int = 10) -> Dict:
        """Get explanation for the prediction"""
        try:
            # Simple word-based explanation (fallback when SHAP not available)
            tokens = self.tokenizer.tokenize(text)
            
            # Create a simple word-based explanation
            word_importance = []
            for token in tokens:
                # Simple heuristic: longer words and certain patterns are more important
                importance = len(token) * 0.1
                if token.lower() in ['memory', 'remember', 'forget', 'confused', 'trouble', 'difficult']:
                    importance += 0.5
                elif token.lower() in ['well', 'good', 'fine', 'okay', 'normal', 'clear']:
                    importance -= 0.3
                word_importance.append((token, importance))
            
            # Sort by importance
            word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            top_tokens = word_importance[:top_k]
            
            return {
                'top_contributing_tokens': [
                    {'token': token, 'score': float(score)} 
                    for token, score in top_tokens
                ],
                'positive_tokens': [
                    {'token': token, 'score': float(score)} 
                    for token, score in top_tokens if score > 0
                ],
                'negative_tokens': [
                    {'token': token, 'score': float(score)} 
                    for token, score in top_tokens if score < 0
                ],
                'note': 'Using simplified word-based explanation'
            }
        except Exception as e:
            return {'error': f'Could not generate explanation: {str(e)}'}
    
    def batch_predict(self, texts: List[str], return_features: bool = False) -> List[Dict]:
        """Predict for multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text, return_features=return_features)
            results.append(result)
        return results
    
    def evaluate_on_dataset(self, df: pd.DataFrame, text_column: str = 'text', 
                           label_column: str = 'label') -> Dict:
        """Evaluate model on a dataset"""
        predictions = []
        true_labels = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            true_label = row[label_column]
            
            result = self.predict(text)
            predictions.append(result['probability_ad'])
            true_labels.append(true_label)
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'auc': roc_auc_score(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        return metrics
    
    def save_prediction_report(self, results: List[Dict], output_path: str):
        """Save prediction results to file"""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Prediction report saved to: {output_path}")


def main():
    """Example usage of the inference class"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AD Detection Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--calibration_path', type=str, default=None,
                       help='Path to calibration model')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze')
    parser.add_argument('--input_file', type=str, default=None,
                       help='File containing texts to analyze')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results')
    parser.add_argument('--explain', action='store_true',
                       help='Generate explanations')
    
    args = parser.parse_args()
    
    # Initialize inference model
    inference = ADInference(
        model_path=args.model_path,
        calibration_path=args.calibration_path
    )
    
    if args.text:
        # Single text prediction
        result = inference.predict(
            args.text, 
            return_features=True, 
            return_explanation=args.explain
        )
        
        print("\nPrediction Result:")
        print(f"Text: {args.text[:100]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"AD Probability: {result['probability_ad']:.3f}")
        print(f"Normal Probability: {result['probability_normal']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if 'explanation' in result and result['explanation']:
            print("\nTop Contributing Tokens:")
            for token_info in result['explanation']['top_contributing_tokens'][:5]:
                print(f"  {token_info['token']}: {token_info['score']:.3f}")
    
    elif args.input_file:
        # Batch prediction
        df = pd.read_csv(args.input_file)
        results = inference.batch_predict(
            df['text'].tolist(), 
            return_features=True
        )
        
        # Save results
        if args.output_file:
            inference.save_prediction_report(results, args.output_file)
        else:
            print(f"Processed {len(results)} texts")
            print(f"AD predictions: {sum(1 for r in results if r['prediction'] == 'AD')}")
            print(f"Normal predictions: {sum(1 for r in results if r['prediction'] == 'Normal')}")
    
    else:
        print("Please provide either --text or --input_file argument")


if __name__ == "__main__":
    main()