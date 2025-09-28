"""
Training script for AD detection using RoBERTa with cross-validation
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import Dataset as HFDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, DATA_CONFIG, EVAL_CONFIG, MODELS_DIR, RESULTS_DIR, LOGS_DIR
from data_preprocessing import TranscriptPreprocessor


class ADDataset(Dataset):
    """Custom dataset for AD detection"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ADTrainer:
    """Main trainer class for AD detection model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.cv_results = []
        self.best_model_path = None
        
    def setup_model(self):
        """Initialize tokenizer and model"""
        print(f"Loading tokenizer and model: {self.config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=self.config['num_labels']
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate AUC using probabilities
        probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=-1)[:, 1].numpy()
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5  # Default value when only one class present
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': auc
        }
    
    def train_fold(self, train_dataset, val_dataset, fold: int) -> Dict:
        """Train model for a single fold"""
        print(f"\nTraining fold {fold + 1}/{self.config.get('cv_folds', 5)}")
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=f"{MODELS_DIR}/fold_{fold}",
            eval_strategy=self.config['evaluation_strategy'],  # Updated parameter name
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['eval_batch_size'],
            num_train_epochs=self.config['num_epochs'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            fp16=self.config['fp16'],
            save_strategy=self.config['save_strategy'],
            load_best_model_at_end=self.config['load_best_model_at_end'],
            metric_for_best_model=self.config['metric_for_best_model'],
            save_total_limit=self.config['save_total_limit'],
            logging_dir=f"{LOGS_DIR}/fold_{fold}",
            logging_steps=50,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate on validation set
        eval_result = trainer.evaluate()
        
        # Get predictions for calibration
        val_predictions = trainer.predict(val_dataset)
        val_probs = torch.softmax(torch.tensor(val_predictions.predictions), dim=-1)[:, 1].numpy()
        val_labels = val_predictions.label_ids
        
        # Train calibration model
        calibration_model = self._train_calibration_model(val_probs, val_labels)
        
        # Save model and calibration
        fold_model_path = f"{MODELS_DIR}/fold_{fold}_final"
        trainer.save_model(fold_model_path)
        self.tokenizer.save_pretrained(fold_model_path)
        
        with open(f"{fold_model_path}/calibration_model.pkl", 'wb') as f:
            pickle.dump(calibration_model, f)
        
        return {
            'fold': fold,
            'train_loss': train_result.training_loss,
            'eval_metrics': eval_result,
            'val_probs': val_probs,
            'val_labels': val_labels,
            'model_path': fold_model_path
        }
    
    def _train_calibration_model(self, probs: np.ndarray, labels: np.ndarray):
        """Train calibration model using isotonic regression"""
        if self.config.get('calibration_method', 'isotonic') == 'isotonic':
            calibration_model = IsotonicRegression(out_of_bounds='clip')
        else:  # Platt scaling
            calibration_model = LogisticRegression()
        
        calibration_model.fit(probs.reshape(-1, 1), labels)
        return calibration_model
    
    def cross_validate(self, df: pd.DataFrame) -> Dict:
        """Perform cross-validation training"""
        print("Starting cross-validation training...")
        
        # Prepare data
        texts = df[self.config['text_column']].tolist()
        labels = df[self.config['label_column']].tolist()
        
        # Check class distribution
        label_counts = pd.Series(labels).value_counts()
        min_class_count = label_counts.min()
        requested_folds = self.config.get('cv_folds', 5)
        
        # Adjust number of folds based on smallest class
        actual_folds = min(requested_folds, min_class_count)
        if actual_folds < requested_folds:
            print(f"Warning: Reducing CV folds from {requested_folds} to {actual_folds} due to small class size")
            print(f"Class distribution: {label_counts.to_dict()}")
        
        # If we have very few samples, use a simple train/val split instead
        if actual_folds < 2:
            print("Warning: Too few samples for cross-validation. Using simple train/validation split instead.")
            return self._simple_train_val_split(df, texts, labels)
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=actual_folds,
            shuffle=True,
            random_state=self.config.get('random_state', 42)
        )
        
        self.cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            # Create datasets for this fold
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            train_dataset = ADDataset(train_texts, train_labels, self.tokenizer, self.config['max_length'])
            val_dataset = ADDataset(val_texts, val_labels, self.tokenizer, self.config['max_length'])
            
            # Train fold
            fold_result = self.train_fold(train_dataset, val_dataset, fold)
            self.cv_results.append(fold_result)
            
            # Clear GPU memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics()
        
        # Select best model (highest average AUC)
        best_fold = max(self.cv_results, key=lambda x: x['eval_metrics']['roc_auc'])
        self.best_model_path = best_fold['model_path']
        
        print(f"\nCross-validation completed!")
        print(f"Average metrics: {avg_metrics}")
        print(f"Best model saved at: {self.best_model_path}")
        
        return {
            'cv_results': self.cv_results,
            'average_metrics': avg_metrics,
            'best_model_path': self.best_model_path
        }
    
    def _simple_train_val_split(self, df: pd.DataFrame, texts: List[str], labels: List[int]) -> Dict:
        """Simple train/validation split when cross-validation is not possible"""
        from sklearn.model_selection import train_test_split
        
        # Split the data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ADDataset(train_texts, train_labels, self.tokenizer, self.config['max_length'])
        val_dataset = ADDataset(val_texts, val_labels, self.tokenizer, self.config['max_length'])
        
        # Train single fold
        fold_result = self.train_fold(train_dataset, val_dataset, 0)
        
        # Calculate average metrics (same as single fold)
        avg_metrics = self._calculate_average_metrics()
        
        # Select best model
        self.best_model_path = fold_result['model_path']
        
        print(f"\nSimple train/val split completed!")
        print(f"Average metrics: {avg_metrics}")
        print(f"Best model saved at: {self.best_model_path}")
        
        return {
            'cv_results': [fold_result],
            'average_metrics': avg_metrics,
            'best_model_path': self.best_model_path
        }
    
    def _calculate_average_metrics(self) -> Dict:
        """Calculate average metrics across folds"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        avg_metrics = {}
        
        for metric in metrics:
            values = [result['eval_metrics'][metric] for result in self.cv_results]
            avg_metrics[f'{metric}_mean'] = np.mean(values)
            avg_metrics[f'{metric}_std'] = np.std(values)
        
        return avg_metrics
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save training results"""
        if output_path is None:
            output_path = f"{RESULTS_DIR}/training_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if key == 'cv_results':
                serializable_results[key] = []
                for fold_result in value:
                    serializable_fold = {}
                    for k, v in fold_result.items():
                        if isinstance(v, np.ndarray):
                            serializable_fold[k] = v.tolist()
                        else:
                            serializable_fold[k] = v
                    serializable_results[key].append(serializable_fold)
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    def plot_training_curves(self, output_path: str = None):
        """Plot training curves and metrics"""
        if output_path is None:
            output_path = f"{RESULTS_DIR}/training_curves.png"
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Cross-Validation Results', fontsize=16)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            
            values = [result['eval_metrics'][metric] for result in self.cv_results]
            axes[row, col].bar(range(1, len(values) + 1), values)
            axes[row, col].set_title(f'{metric.upper()}')
            axes[row, col].set_xlabel('Fold')
            axes[row, col].set_ylabel(metric)
            axes[row, col].set_ylim(0, 1)
            
            # Add mean line
            mean_val = np.mean(values)
            axes[row, col].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7)
            axes[row, col].text(0.5, 0.95, f'Mean: {mean_val:.3f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top', ha='center')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {output_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AD detection model')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations CSV file')
    parser.add_argument('--transcripts_dir', type=str, required=True,
                       help='Directory containing transcript files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = {**MODEL_CONFIG, **DATA_CONFIG, **EVAL_CONFIG}
    config['cv_folds'] = args.cv_folds
    
    if args.output_dir:
        global RESULTS_DIR
        RESULTS_DIR = Path(args.output_dir)
        RESULTS_DIR.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = TranscriptPreprocessor(config)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = preprocessor.load_data(args.annotations, args.transcripts_dir)
    df_processed = preprocessor.preprocess_dataset(df)
    
    print(f"Dataset shape: {df_processed.shape}")
    print(f"Class distribution: {df_processed[config['label_column']].value_counts()}")
    
    # Initialize trainer
    trainer = ADTrainer(config)
    trainer.setup_model()
    
    # Perform cross-validation
    results = trainer.cross_validate(df_processed)
    
    # Save results
    trainer.save_results(results)
    trainer.plot_training_curves()
    
    print("\nTraining completed successfully!")
    print(f"Best model saved at: {trainer.best_model_path}")


if __name__ == "__main__":
    main()