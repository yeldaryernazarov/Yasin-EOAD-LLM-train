# Alzheimer's Disease Detection from Speech Transcripts

A comprehensive machine learning pipeline for detecting Alzheimer's Disease (AD) from speech transcripts using RoBERTa fine-tuning with textual and paralinguistic features.

## Features

- **Text Preprocessing**: Advanced text cleaning and normalization
- **Feature Extraction**: Linguistic and paralinguistic features
- **Model Training**: RoBERTa fine-tuning with cross-validation
- **Evaluation**: Comprehensive metrics and calibration
- **Inference**: Real-time prediction with confidence scores
- **Explainability**: SHAP-based model interpretation
- **Demographic Analysis**: Stratified evaluation by age, gender, etc.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Yasin-EOAD-LLM-train.git
cd Yasin-EOAD-LLM-train
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data (done automatically on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

## Data Format

Your data should be organized as follows:

### Annotations CSV
```csv
id,age,gender,label,text_path
1,75,M,1,transcript_001.txt
2,68,F,0,transcript_002.txt
3,72,M,1,transcript_003.txt
...
```

Where:
- `id`: Unique identifier
- `age`: Patient age
- `gender`: Patient gender (M/F)
- `label`: 0 for normal, 1 for AD
- `text_path`: Path to transcript file (relative to transcripts directory)

### Transcript Files
Plain text files (`.txt`) containing speech transcripts.

## Quick Start

### 1. Run Full Pipeline
```bash
python main.py pipeline --annotations data/annotations.csv --transcripts_dir data/transcripts/
```

### 2. Train Model Only
```bash
python main.py train --annotations data/annotations.csv --transcripts_dir data/transcripts/ --cv_folds 5
```

### 3. Evaluate Model
```bash
python main.py evaluate --model_path models/best_model --data_file data/test_data.csv
```

### 4. Make Predictions
```bash
# Single text
python main.py infer --model_path models/best_model --text "Hello, how are you today?"

# Batch prediction
python main.py infer --model_path models/best_model --input_file data/test_texts.csv --output_file results/predictions.csv
```

### 5. Explain Predictions
```bash
# Single text explanation
python main.py explain --model_path models/best_model --text "Hello, how are you today?"

# Batch analysis
python main.py explain --model_path models/best_model --data_file data/test_data.csv --sample_size 100
```

## Detailed Usage

### Data Preprocessing

The preprocessing pipeline extracts both linguistic and paralinguistic features:

**Linguistic Features:**
- Basic statistics (word count, sentence count, etc.)
- Readability scores (Flesch-Kincaid, SMOG, etc.)
- POS tag ratios (nouns, verbs, adjectives, adverbs)
- Lexical diversity (type-token ratio)
- Dysfluency indicators (um/uh counts, repetitions)

**Paralinguistic Features:**
- Pause patterns
- Question/exclamation ratios
- Hesitation markers
- Repetition patterns
- Sentence complexity
- Vocabulary richness

### Model Training

The training process includes:
- 5-fold stratified cross-validation
- RoBERTa-base fine-tuning
- Early stopping
- Model calibration
- Comprehensive evaluation metrics

### Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Brier Score**: Calibration quality

### Explainability

The system provides SHAP-based explanations showing:
- Most important tokens contributing to predictions
- Positive and negative feature contributions
- Feature importance across the dataset
- Word clouds of important features

## Configuration

Edit `config.py` to customize:

- Model parameters (learning rate, batch size, etc.)
- Feature extraction settings
- Evaluation metrics
- Inference thresholds

## File Structure

```
Yasin-EOAD-LLM-train/
├── main.py                 # Main pipeline script
├── config.py              # Configuration settings
├── data_preprocessing.py   # Data loading and preprocessing
├── train_model.py         # Model training with cross-validation
├── evaluate_model.py      # Model evaluation and calibration
├── inference.py           # Real-time inference
├── explainability.py      # SHAP-based explanations
├── requirements.txt       # Dependencies
├── README.md             # This file
├── data/                 # Data directory
│   ├── annotations.csv   # Your annotations file
│   └── transcripts/      # Your transcript files
├── models/               # Trained models
├── results/              # Results and plots
└── logs/                 # Training logs
```

## Example Python Usage

```python
from main import ADPipeline

# Initialize pipeline
pipeline = ADPipeline()

# Setup data
df = pipeline.setup_data('data/annotations.csv', 'data/transcripts/')

# Train model
model_path = pipeline.train_model(df, cv_folds=5)

# Setup inference
pipeline.setup_inference(model_path)

# Make prediction
result = pipeline.predict("Hello, how are you today?", explain=True)
print(f"Prediction: {result['prediction']}")
print(f"AD Probability: {result['probability_ad']:.3f}")

# Get explanation
explanation = pipeline.explain_prediction("Hello, how are you today?")
print(f"Top features: {explanation['top_features'][:5]}")
```

## Advanced Features

### Demographic Analysis
The system can analyze performance across different demographic groups:
```python
# Evaluate by age and gender
evaluator = ADModelEvaluator(model_path)
report = evaluator.generate_evaluation_report(
    df, 
    demographic_columns=['age', 'gender']
)
```

### Model Calibration
Improve probability calibration:
```python
# Train calibration model
calibration_results = evaluator.train_calibration_model(
    df, method='isotonic'
)
```

### Custom Feature Extraction
Add your own features by extending the `TranscriptPreprocessor` class:
```python
class CustomPreprocessor(TranscriptPreprocessor):
    def extract_custom_features(self, text):
        # Your custom feature extraction
        return features
```

## Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory Management**: Adjust batch size based on available memory
3. **Data Quality**: Clean and preprocess your data thoroughly
4. **Cross-Validation**: Use sufficient folds for reliable evaluation
5. **Feature Selection**: Consider feature importance for model optimization

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `config.py`
2. **Empty Transcripts**: Check file paths and encoding
3. **Poor Performance**: Ensure balanced dataset and sufficient data
4. **SHAP Errors**: Install latest version: `pip install shap>=0.42.0`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ad_detection_pipeline,
  title={Alzheimer's Disease Detection from Speech Transcripts},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and support, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

- Hugging Face Transformers library
- SHAP for explainability
- NLTK for text processing
- Scikit-learn for evaluation metrics