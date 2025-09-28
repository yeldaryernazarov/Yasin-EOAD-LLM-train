"""
Script to download required NLTK data
"""
import nltk

def download_nltk_data():
    """Download all required NLTK data"""
    print("Downloading NLTK data...")
    
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
                print(f"  {data}: already downloaded")
            elif data == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
                print(f"  {data}: already downloaded")
            elif data == 'stopwords':
                nltk.data.find('corpora/stopwords')
                print(f"  {data}: already downloaded")
            elif data == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
                print(f"  {data}: already downloaded")
            elif data == 'maxent_ne_chunker':
                nltk.data.find('chunkers/maxent_ne_chunker')
                print(f"  {data}: already downloaded")
            elif data == 'words':
                nltk.data.find('corpora/words')
                print(f"  {data}: already downloaded")
        except LookupError:
            print(f"  Downloading {data}...")
            nltk.download(data)
            print(f"  {data}: downloaded")
    
    print("NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data()