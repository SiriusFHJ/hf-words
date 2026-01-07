import os
import time
import re
import sys
from collections import Counter
from typing import List, Tuple, Generator

import pandas as pd
import spacy
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

# Configuration
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

# Hyperparameters & Settings
TOP_UNIGRAMS = int(os.getenv("TOP_UNIGRAMS", 1000))
TOP_BIGRAMS = int(os.getenv("TOP_BIGRAMS", 100))
TOP_TRIGRAMS = int(os.getenv("TOP_TRIGRAMS", 100))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", 10000))
DATA_NAME = os.getenv("DATA_NAME", "wikitext")
MIN_FREQUENCY = int(os.getenv("MIN_FREQUENCY", 20))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 200)) # Smaller batch size for Transformer models on GPU

def get_device_info():
    """Check for GPU availability and setup spaCy."""
    if torch.cuda.is_available():
        try:
            spacy.require_gpu()
            print(f"🚀 GPU Acceleration Enabled: {torch.cuda.get_device_name(0)}")
            return True
        except Exception as e:
            print(f"⚠️ GPU detected but spaCy failed to use it: {e}")
            return False
    else:
        print("⚠️ No GPU detected. Running on CPU (this might be slow for the GPU script).")
        return False

def load_spacy_model(use_gpu: bool):
    """
    Load spaCy model. 
    Using 'en_core_web_trf' for transformer-based GPU acceleration.
    """
    model_name = "en_core_web_trf"
    try:
        # Disable components we don't need for simple lemma extraction
        # For TRF models, 'tagger' and 'attribute_ruler' are usually needed for lemmas.
        # 'transformer' is essential.
        # We can disable 'ner', 'parser', 'textcat' safely.
        print(f"Loading {model_name}...")
        nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])
        return nlp
    except OSError:
        print(f"Model '{model_name}' not found.")
        print(f"Please run: python -m spacy download {model_name}")
        print("Note: You also need spacy-transformers installed: pip install spacy-transformers")
        sys.exit(1)

def text_generator(dataset, limit: int) -> Generator[str, None, None]:
    """Yields text from the dataset."""
    count = 0
    # dataset is iterable
    for example in dataset:
        if count >= limit:
            break
        text = example.get('text', '')
        if text.strip():
            yield text
            count += 1

def build_custom_vocab_gpu():
    use_gpu = get_device_info()
    nlp = load_spacy_model(use_gpu)
    
    print(f"Dataset: {DATA_NAME}")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # Load streaming dataset
    print("Loading dataset stream...")
    dataset = load_dataset(DATA_NAME, split="train", streaming=True)
    
    # Initialize global counters
    total_unigrams = Counter()
    total_bigrams = Counter()
    total_trigrams = Counter()
    
    start_time = time.time()
    
    # Prepare generator
    data_gen = text_generator(dataset, SAMPLE_SIZE)
    
    print("Processing pipeline started...")
    
    # Use nlp.pipe for efficient batch processing on GPU
    # n_process=1 is recommended for GPU to avoid context switching/spawn overhead
    doc_stream = nlp.pipe(data_gen, batch_size=BATCH_SIZE, n_process=1)
    
    # Process stream
    # We iterate manually to update progress bar
    for doc in tqdm(doc_stream, total=SAMPLE_SIZE, unit="doc"):
        words = [
            token.lemma_ for token in doc 
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
        ]
        
        if not words:
            continue
            
        total_unigrams.update(words)
        
        if len(words) >= 2:
            bigrams = [" ".join(words[j:j+2]) for j in range(len(words)-1)]
            total_bigrams.update(bigrams)
        
        if len(words) >= 3:
            trigrams = [" ".join(words[j:j+3]) for j in range(len(words)-2)]
            total_trigrams.update(trigrams)

    process_time = time.time() - start_time
    print(f"\nProcessing complete in {process_time:.2f} seconds ({SAMPLE_SIZE/process_time:.1f} docs/s)")
    
    # Convert to DataFrames and Save
    print("Saving results...")
    
    # Unigrams
    df_unigrams = pd.DataFrame(total_unigrams.most_common(TOP_UNIGRAMS), columns=['Word', 'Frequency'])
    df_unigrams.to_csv("words.csv", index=False)
    
    # Bigrams
    df_bigrams = pd.DataFrame(total_bigrams.most_common(TOP_BIGRAMS), columns=['Phrase', 'Frequency'])
    df_bigrams = df_bigrams[df_bigrams['Frequency'] > MIN_FREQUENCY]
    df_bigrams.to_csv("phrases.csv", index=False)
    
    # Trigrams
    df_trigrams = pd.DataFrame(total_trigrams.most_common(TOP_TRIGRAMS), columns=['Trigram', 'Frequency'])
    df_trigrams = df_trigrams[df_trigrams['Frequency'] > MIN_FREQUENCY]
    df_trigrams.to_csv("trigrams.csv", index=False)
    
    print("✅ Done! Saved to words.csv, phrases.csv, trigrams.csv")

if __name__ == "__main__":
    build_custom_vocab_gpu()
