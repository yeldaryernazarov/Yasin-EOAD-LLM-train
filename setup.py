"""
Setup script for AD detection pipeline
"""
import os
import sys
import subprocess
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/transcripts", 
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        
        nltk_data = [
            'punkt',
            'punkt_tab',  # New requirement for newer NLTK versions
            'stopwords', 
            'averaged_perceptron_tagger',
            'maxent_ne_chunker',
            'words'
        ]
        
        for data in nltk_data:
            try:
                if data == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                elif data == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif data == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                elif data == 'maxent_ne_chunker':
                    nltk.data.find('chunkers/maxent_ne_chunker')
                elif data == 'words':
                    nltk.data.find('corpora/words')
                print(f"  {data}: already downloaded")
            except LookupError:
                nltk.download(data)
                print(f"  {data}: downloaded")
        
        print("NLTK data downloaded successfully!")
    except ImportError:
        print("NLTK not available. Please install it first.")
        return False
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False
    return True


def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    # Sample annotations
    annotations_data = [
        {"id": 1, "age": 75, "gender": "M", "label": 0, "text_path": "transcript_001.txt"},
        {"id": 2, "age": 68, "gender": "F", "label": 0, "text_path": "transcript_002.txt"},
        {"id": 3, "age": 72, "gender": "M", "label": 1, "text_path": "transcript_003.txt"},
        {"id": 4, "age": 80, "gender": "F", "label": 1, "text_path": "transcript_004.txt"},
        {"id": 5, "age": 70, "gender": "M", "label": 0, "text_path": "transcript_005.txt"},
    ]
    
    # Sample transcript texts
    transcript_texts = [
        "Hello, how are you today? I'm feeling quite well, thank you for asking. The weather is nice.",
        "I went to the store yesterday and bought some groceries. It was a pleasant shopping trip.",
        "I'm having trouble remembering things lately. Sometimes I forget where I put my keys.",
        "What was I saying? Oh yes, I was talking about my doctor's appointment. I can't remember the details.",
        "I don't remember what I had for breakfast this morning. It's getting harder to recall things.",
    ]
    
    # Save annotations
    import pandas as pd
    annotations_df = pd.DataFrame(annotations_data)
    annotations_df.to_csv("data/annotations.csv", index=False)
    
    # Save transcript files
    for i, text in enumerate(transcript_texts, 1):
        with open(f"data/transcripts/transcript_{i:03d}.txt", "w", encoding="utf-8") as f:
            f.write(text)
    
    print("Sample data created successfully!")
    print("  - data/annotations.csv")
    print("  - data/transcripts/transcript_*.txt")


def check_gpu():
    """Check if GPU is available"""
    print("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  GPU not available. Training will use CPU (slower).")
    except ImportError:
        print("  PyTorch not installed yet.")


def main():
    """Main setup function"""
    print("Setting up AD Detection Pipeline")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return
    
    # Download NLTK data
    print("\n3. Downloading NLTK data...")
    if not download_nltk_data():
        print("Failed to download NLTK data. Please install NLTK manually.")
    
    # Check GPU
    print("\n4. Checking GPU...")
    check_gpu()
    
    # Create sample data
    print("\n5. Creating sample data...")
    create_sample_data()
    
    print("\n" + "=" * 40)
    print("SETUP COMPLETED!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Run the example: python example_usage.py")
    print("2. Or run the full pipeline: python main.py pipeline --annotations data/annotations.csv --transcripts_dir data/transcripts/")
    print("3. Check the README.md for detailed usage instructions")
    
    print("\nYour data structure:")
    print("data/")
    print("├── annotations.csv")
    print("└── transcripts/")
    print("    ├── transcript_001.txt")
    print("    ├── transcript_002.txt")
    print("    └── ...")
    
    print("\nTo use your own data:")
    print("1. Replace data/annotations.csv with your annotations")
    print("2. Replace data/transcripts/*.txt with your transcript files")
    print("3. Update the column names in config.py if needed")


if __name__ == "__main__":
    main()