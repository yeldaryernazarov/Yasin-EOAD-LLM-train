"""
Data preprocessing pipeline for AD detection
Handles loading, cleaning, and feature extraction from transcripts
"""
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data"""
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
        except LookupError:
            print(f"Downloading NLTK data: {data}")
            nltk.download(data)

# Download NLTK data
download_nltk_data()


class TranscriptPreprocessor:
    """Preprocesses transcript data for AD detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        
    def load_data(self, annotations_path: str, transcripts_dir: str) -> pd.DataFrame:
        """
        Load annotations and corresponding transcript files
        
        Args:
            annotations_path: Path to CSV file with annotations
            transcripts_dir: Directory containing transcript files
            
        Returns:
            DataFrame with loaded data
        """
        # Load annotations
        df = pd.read_csv(annotations_path)
        
        # Load transcript texts
        texts = []
        for idx, row in df.iterrows():
            text_path = os.path.join(transcripts_dir, row[self.config['text_path_column']])
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                texts.append(text)
            except FileNotFoundError:
                print(f"Warning: Transcript file not found: {text_path}")
                texts.append("")
            except Exception as e:
                print(f"Error reading {text_path}: {e}")
                texts.append("")
        
        df[self.config['text_column']] = texts
        return df
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Expand common contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'s": " is", "'ve": " have",
            "'ll": " will", "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text.strip()
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text
        
        Args:
            text: Cleaned text
            
        Returns:
            Dictionary of linguistic features
        """
        if not text or text.strip() == "":
            return {key: 0.0 for key in self._get_linguistic_feature_names()}
        
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Safe sentence tokenization
        try:
            features['sentence_count'] = len(sent_tokenize(text))
        except LookupError:
            # Fallback to simple sentence counting if NLTK data not available
            features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Average sentence length
        try:
            sentences = sent_tokenize(text)
            features['avg_sentence_length'] = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
        except LookupError:
            # Fallback calculation
            sentence_count = features['sentence_count']
            features['avg_sentence_length'] = features['word_count'] / sentence_count if sentence_count > 0 else 0
        
        # Type-token ratio (lexical diversity)
        unique_words = set(words)
        features['type_token_ratio'] = len(unique_words) / len(words) if words else 0
        
        # Readability scores
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        features['smog_index'] = textstat.smog_index(text)
        
        # POS tag analysis
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            pos_counts = {}
            for word, tag in pos_tags:
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
            
            total_tokens = len(tokens)
            if total_tokens > 0:
                features['noun_ratio'] = pos_counts.get('NN', 0) / total_tokens
                features['verb_ratio'] = pos_counts.get('VB', 0) / total_tokens
                features['adj_ratio'] = pos_counts.get('JJ', 0) / total_tokens
                features['adv_ratio'] = pos_counts.get('RB', 0) / total_tokens
            else:
                features['noun_ratio'] = 0
                features['verb_ratio'] = 0
                features['adj_ratio'] = 0
                features['adv_ratio'] = 0
        except (LookupError, Exception):
            # Fallback to simple word pattern matching
            words = text.split()
            total_words = len(words)
            if total_words > 0:
                # Simple heuristics for POS-like features
                noun_patterns = ['tion', 'sion', 'ness', 'ment', 'ity']
                verb_patterns = ['ing', 'ed', 'en']
                adj_patterns = ['ful', 'less', 'ous', 'ive']
                adv_patterns = ['ly']
                
                noun_count = sum(1 for word in words if any(pattern in word.lower() for pattern in noun_patterns))
                verb_count = sum(1 for word in words if any(pattern in word.lower() for pattern in verb_patterns))
                adj_count = sum(1 for word in words if any(pattern in word.lower() for pattern in adj_patterns))
                adv_count = sum(1 for word in words if any(pattern in word.lower() for pattern in adv_patterns))
                
                features['noun_ratio'] = noun_count / total_words
                features['verb_ratio'] = verb_count / total_words
                features['adj_ratio'] = adj_count / total_words
                features['adv_ratio'] = adv_count / total_words
            else:
                features['noun_ratio'] = 0
                features['verb_ratio'] = 0
                features['adj_ratio'] = 0
                features['adv_ratio'] = 0
        
        # Dysfluency indicators
        features['um_count'] = text.lower().count(' um ') + text.lower().count(' uh ')
        features['repetition_ratio'] = self._calculate_repetition_ratio(text)
        
        return features
    
    def extract_paralinguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract paralinguistic features from text
        
        Args:
            text: Cleaned text
            
        Returns:
            Dictionary of paralinguistic features
        """
        if not text or text.strip() == "":
            return {key: 0.0 for key in self._get_paralinguistic_feature_names()}
        
        features = {}
        
        # Pause indicators (based on punctuation patterns)
        features['pause_ratio'] = (text.count('.') + text.count('!') + text.count('?')) / len(text.split()) if text.split() else 0
        
        # Question patterns
        try:
            sentences = sent_tokenize(text)
            features['question_ratio'] = text.count('?') / len(sentences) if sentences else 0
        except LookupError:
            features['question_ratio'] = text.count('?') / max(features['sentence_count'], 1)
        
        # Exclamation patterns
        try:
            sentences = sent_tokenize(text)
            features['exclamation_ratio'] = text.count('!') / len(sentences) if sentences else 0
        except LookupError:
            features['exclamation_ratio'] = text.count('!') / max(features['sentence_count'], 1)
        
        # Hesitation markers
        hesitation_markers = ['um', 'uh', 'er', 'ah', 'well', 'like', 'you know']
        features['hesitation_ratio'] = sum(text.lower().count(marker) for marker in hesitation_markers) / len(text.split()) if text.split() else 0
        
        # Repetition patterns
        features['word_repetition_ratio'] = self._calculate_word_repetition_ratio(text)
        features['phrase_repetition_ratio'] = self._calculate_phrase_repetition_ratio(text)
        
        # Sentence complexity
        try:
            sentences = sent_tokenize(text)
            if sentences:
                complex_sentences = sum(1 for sent in sentences if len(sent.split()) > 15)
                features['complex_sentence_ratio'] = complex_sentences / len(sentences)
            else:
                features['complex_sentence_ratio'] = 0
        except LookupError:
            # Fallback calculation
            sentence_count = features['sentence_count']
            if sentence_count > 0:
                # Simple heuristic: sentences with more than 15 words
                words = text.split()
                avg_words_per_sentence = len(words) / sentence_count
                features['complex_sentence_ratio'] = 1.0 if avg_words_per_sentence > 15 else 0.0
            else:
                features['complex_sentence_ratio'] = 0
        
        # Vocabulary richness
        words = text.split()
        if words:
            features['vocabulary_richness'] = len(set(words)) / len(words)
        else:
            features['vocabulary_richness'] = 0
        
        return features
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated words"""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        repeated_words = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_words += 1
        
        return repeated_words / len(words)
    
    def _calculate_word_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated words in sequence"""
        words = text.split()
        if len(words) < 2:
            return 0.0
        
        consecutive_repeats = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                consecutive_repeats += 1
        
        return consecutive_repeats / len(words)
    
    def _calculate_phrase_repetition_ratio(self, text: str) -> float:
        """Calculate ratio of repeated phrases"""
        words = text.split()
        if len(words) < 4:
            return 0.0
        
        # Look for 2-word phrase repetitions
        phrases = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        phrase_counts = {}
        for phrase in phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
        return repeated_phrases / len(phrases)
    
    def _get_linguistic_feature_names(self) -> List[str]:
        """Get list of linguistic feature names"""
        return [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'type_token_ratio', 'flesch_reading_ease',
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'noun_ratio',
            'verb_ratio', 'adj_ratio', 'adv_ratio', 'um_count', 'repetition_ratio'
        ]
    
    def _get_paralinguistic_feature_names(self) -> List[str]:
        """Get list of paralinguistic feature names"""
        return [
            'pause_ratio', 'question_ratio', 'exclamation_ratio', 'hesitation_ratio',
            'word_repetition_ratio', 'phrase_repetition_ratio', 'complex_sentence_ratio',
            'vocabulary_richness'
        ]
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire dataset
        
        Args:
            df: Raw dataset
            
        Returns:
            Preprocessed dataset with features
        """
        print("Cleaning texts...")
        df[self.config['text_column']] = df[self.config['text_column']].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df[self.config['text_column']].str.len() > 0].reset_index(drop=True)
        
        if self.config['linguistic_features']:
            print("Extracting linguistic features...")
            linguistic_features = df[self.config['text_column']].apply(self.extract_linguistic_features)
            linguistic_df = pd.DataFrame(linguistic_features.tolist())
            df = pd.concat([df, linguistic_df], axis=1)
        
        if self.config['paralinguistic_features']:
            print("Extracting paralinguistic features...")
            paralinguistic_features = df[self.config['text_column']].apply(self.extract_paralinguistic_features)
            paralinguistic_df = pd.DataFrame(paralinguistic_features.tolist())
            df = pd.concat([df, paralinguistic_df], axis=1)
        
        return df
    
    def create_tfidf_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, TfidfVectorizer]:
        """
        Create TF-IDF features from text
        
        Args:
            df: Dataset with text column
            
        Returns:
            TF-IDF features and vectorizer
        """
        vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=self.config['ngram_range'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df'],
            stop_words='english'
        )
        
        tfidf_features = vectorizer.fit_transform(df[self.config['text_column']])
        return tfidf_features.toarray(), vectorizer


def main():
    """Example usage of the preprocessor"""
    from config import DATA_CONFIG, FEATURE_CONFIG
    
    # Merge configs
    config = {**DATA_CONFIG, **FEATURE_CONFIG}
    
    preprocessor = TranscriptPreprocessor(config)
    
    # Example usage (replace with your actual paths)
    # df = preprocessor.load_data("data/annotations.csv", "data/transcripts/")
    # df_processed = preprocessor.preprocess_dataset(df)
    # print(f"Processed dataset shape: {df_processed.shape}")
    # print(f"Features: {list(df_processed.columns)}")


if __name__ == "__main__":
    main()