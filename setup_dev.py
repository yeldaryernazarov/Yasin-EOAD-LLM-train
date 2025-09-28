#!/usr/bin/env python3
"""
Development setup script for Alzheimer's Disease Detection Project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up development environment for Alzheimer's Disease Detection Project")
    print("=" * 70)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: You're not in a virtual environment!")
        print("   It's recommended to create one first:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Setup cancelled.")
            return
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return
    
    # Install main dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing main dependencies"):
        return
    
    # Install development dependencies
    if not run_command(f"{sys.executable} -m pip install -r requirements-dev.txt", "Installing development dependencies"):
        return
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("   Note: Pre-commit hooks installation failed, but you can continue")
    
    # Download NLTK data
    print("🔄 Downloading NLTK data...")
    try:
        import nltk
        nltk_data = [
            'punkt', 'stopwords', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon'
        ]
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
                print(f"   ✅ Downloaded {data}")
            except Exception as e:
                print(f"   ⚠️  Failed to download {data}: {e}")
        print("✅ NLTK data download completed")
    except ImportError:
        print("   ⚠️  NLTK not available, skipping data download")
    
    # Create necessary directories
    directories = ['data', 'models', 'results', 'logs', 'tests', 'docs', 'examples', 'notebooks']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   📁 Created directory: {directory}")
    
    print("\n" + "=" * 70)
    print("🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Add your data to the 'data/' directory")
    print("2. Run tests: pytest")
    print("3. Run linting: flake8 .")
    print("4. Start development: python main.py --help")
    print("\nHappy coding! 🚀")

if __name__ == "__main__":
    main()
