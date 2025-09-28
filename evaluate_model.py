"""
Comprehensive evaluation and calibration script for AD detection model
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

from config import EVAL_CONFIG, RESULTS_DIR, MODELS_DIR
from data_preprocessing import TranscriptPreprocessor


class ADModelEvaluator:
    """Comprehensive evaluation class for AD detection models"""
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            config: Evaluation configuration
        """
        self.model_path = model_path
        self.config = config or EVAL_CONFIG
        self.model = None
        self.tokenizer = None
        self.calibration_model = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Try to load calibration model
        calibration_path = f"{self.model_path}/calibration_model.pkl"
        if os.path.exists(calibration_path):
            with open(calibration_path, 'rb') as f:
                self.calibration_model = pickle.load(f)
            print("Calibration model loaded")
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities for texts"""
        all_probs = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy()[0])
        
        return np.array(all_probs)
    
    def evaluate_performance(self, df: pd.DataFrame, text_column: str = 'text', 
                           label_column: str = 'label') -> Dict:
        """Evaluate model performance on dataset"""
        print("Evaluating model performance...")
        
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        
        # Get predictions
        probs = self.predict_proba(texts)
        pred_labels = np.argmax(probs, axis=1)
        pred_probs = probs[:, 1]  # Probability of AD class
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        auc = roc_auc_score(true_labels, pred_probs)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Brier score (calibration metric)
        brier_score = brier_score_loss(true_labels, pred_probs)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'auc': auc,
            'ppv': ppv,
            'npv': npv,
            'brier_score': brier_score,
            'confusion_matrix': cm.tolist(),
            'true_positive_rate': sensitivity,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'true_negative_rate': specificity,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        return results
    
    def evaluate_by_demographics(self, df: pd.DataFrame, text_column: str = 'text',
                               label_column: str = 'label', 
                               demographic_columns: List[str] = ['age', 'gender']) -> Dict:
        """Evaluate performance stratified by demographic groups"""
        print("Evaluating performance by demographics...")
        
        results = {}
        
        for col in demographic_columns:
            if col not in df.columns:
                continue
            
            print(f"Evaluating by {col}...")
            col_results = {}
            
            for value in df[col].unique():
                if pd.isna(value):
                    continue
                
                subset = df[df[col] == value]
                if len(subset) < 5:  # Skip groups with too few samples
                    continue
                
                subset_results = self.evaluate_performance(subset, text_column, label_column)
                col_results[str(value)] = subset_results
            
            results[col] = col_results
        
        return results
    
    def plot_roc_curve(self, df: pd.DataFrame, text_column: str = 'text',
                      label_column: str = 'label', output_path: str = None):
        """Plot ROC curve"""
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        probs = self.predict_proba(texts)
        pred_probs = probs[:, 1]
        
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        auc = roc_auc_score(true_labels, pred_probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"ROC curve saved to: {output_path or f'{RESULTS_DIR}/roc_curve.png'}")
    
    def plot_precision_recall_curve(self, df: pd.DataFrame, text_column: str = 'text',
                                   label_column: str = 'label', output_path: str = None):
        """Plot Precision-Recall curve"""
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        probs = self.predict_proba(texts)
        pred_probs = probs[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
        avg_precision = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{RESULTS_DIR}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"Precision-Recall curve saved to: {output_path or f'{RESULTS_DIR}/precision_recall_curve.png'}")
    
    def plot_calibration_curve(self, df: pd.DataFrame, text_column: str = 'text',
                              label_column: str = 'label', output_path: str = None):
        """Plot calibration curve"""
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        probs = self.predict_proba(texts)
        pred_probs = probs[:, 1]
        
        # Plot calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, pred_probs, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Model", color='darkorange')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{RESULTS_DIR}/calibration_curve.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"Calibration curve saved to: {output_path or f'{RESULTS_DIR}/calibration_curve.png'}")
    
    def plot_confusion_matrix(self, df: pd.DataFrame, text_column: str = 'text',
                            label_column: str = 'label', output_path: str = None):
        """Plot confusion matrix"""
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        probs = self.predict_proba(texts)
        pred_labels = np.argmax(probs, axis=1)
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'AD'], 
                   yticklabels=['Normal', 'AD'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"Confusion matrix saved to: {output_path or f'{RESULTS_DIR}/confusion_matrix.png'}")
    
    def train_calibration_model(self, df: pd.DataFrame, text_column: str = 'text',
                              label_column: str = 'label', method: str = 'isotonic') -> Dict:
        """Train calibration model"""
        print(f"Training calibration model using {method}...")
        
        texts = df[text_column].tolist()
        true_labels = df[label_column].tolist()
        probs = self.predict_proba(texts)
        pred_probs = probs[:, 1]
        
        if method == 'isotonic':
            calibration_model = IsotonicRegression(out_of_bounds='clip')
        else:  # Platt scaling
            calibration_model = LogisticRegression()
        
        calibration_model.fit(pred_probs.reshape(-1, 1), true_labels)
        
        # Evaluate calibration
        calibrated_probs = calibration_model.predict_proba(pred_probs.reshape(-1, 1))[:, 1]
        
        # Calculate Brier scores
        original_brier = brier_score_loss(true_labels, pred_probs)
        calibrated_brier = brier_score_loss(true_labels, calibrated_probs)
        
        # Save calibration model
        calibration_path = f"{self.model_path}/calibration_model.pkl"
        with open(calibration_path, 'wb') as f:
            pickle.dump(calibration_model, f)
        
        print(f"Calibration model saved to: {calibration_path}")
        print(f"Original Brier score: {original_brier:.4f}")
        print(f"Calibrated Brier score: {calibrated_brier:.4f}")
        
        return {
            'calibration_model': calibration_model,
            'original_brier_score': original_brier,
            'calibrated_brier_score': calibrated_brier,
            'improvement': original_brier - calibrated_brier
        }
    
    def generate_evaluation_report(self, df: pd.DataFrame, text_column: str = 'text',
                                 label_column: str = 'label', 
                                 demographic_columns: List[str] = None) -> Dict:
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive evaluation report...")
        
        # Overall performance
        performance_results = self.evaluate_performance(df, text_column, label_column)
        
        # Demographic analysis
        demographic_results = {}
        if demographic_columns:
            demographic_results = self.evaluate_by_demographics(
                df, text_column, label_column, demographic_columns
            )
        
        # Generate plots
        self.plot_roc_curve(df, text_column, label_column)
        self.plot_precision_recall_curve(df, text_column, label_column)
        self.plot_calibration_curve(df, text_column, label_column)
        self.plot_confusion_matrix(df, text_column, label_column)
        
        # Compile report
        report = {
            'overall_performance': performance_results,
            'demographic_analysis': demographic_results,
            'model_path': self.model_path,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_size': len(df),
            'class_distribution': df[label_column].value_counts().to_dict()
        }
        
        # Save report
        report_path = f"{RESULTS_DIR}/evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to: {report_path}")
        
        return report


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate AD detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to evaluation data CSV')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column')
    parser.add_argument('--demographic_columns', nargs='+', default=['age', 'gender'],
                       help='Demographic columns for stratified analysis')
    parser.add_argument('--calibrate', action='store_true',
                       help='Train calibration model')
    parser.add_argument('--calibration_method', type=str, default='isotonic',
                       choices=['isotonic', 'platt'],
                       help='Calibration method')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data_file)
    print(f"Loaded evaluation data: {df.shape}")
    print(f"Class distribution: {df[args.label_column].value_counts()}")
    
    # Initialize evaluator
    evaluator = ADModelEvaluator(args.model_path)
    
    # Train calibration if requested
    if args.calibrate:
        calibration_results = evaluator.train_calibration_model(
            df, args.text_column, args.label_column, args.calibration_method
        )
        print(f"Calibration improvement: {calibration_results['improvement']:.4f}")
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report(
        df, args.text_column, args.label_column, args.demographic_columns
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy: {report['overall_performance']['accuracy']:.3f}")
    print(f"Precision: {report['overall_performance']['precision']:.3f}")
    print(f"Recall: {report['overall_performance']['recall']:.3f}")
    print(f"F1-Score: {report['overall_performance']['f1']:.3f}")
    print(f"AUC: {report['overall_performance']['auc']:.3f}")
    print(f"Sensitivity: {report['overall_performance']['sensitivity']:.3f}")
    print(f"Specificity: {report['overall_performance']['specificity']:.3f}")
    print(f"Brier Score: {report['overall_performance']['brier_score']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()