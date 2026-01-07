import pandas as pd
from datasets import load_dataset
from collections import Counter
import spacy
from tqdm import tqdm
from dotenv import load_dotenv
import os
import sys
import concurrent.futures
import re

load_dotenv(override=True)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

USE_SPACY = os.getenv("USE_SPACY", "true").lower() == "true"

# 1. 加载 spaCy 模型（禁用不需要的组件以提高速度）
nlp = None
if USE_SPACY:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
else:
    print("Warning: spaCy is disabled. Using simple regex tokenization.")

TOP_UNIGRAMS = int(os.getenv("TOP_UNIGRAMS"))
TOP_BIGRAMS = int(os.getenv("TOP_BIGRAMS"))
TOP_TRIGRAMS = int(os.getenv("TOP_TRIGRAMS"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE"))
DATA_NAME = os.getenv("DATA_NAME")
MAX_WORKERS = int(os.getenv("MAX_WORKERS")) or os.cpu_count()
MIN_FREQUENCY = int(os.getenv("MIN_FREQUENCY")) or 20

def process_text(text):
    if USE_SPACY and nlp:
        doc = nlp(text.lower())
        words = [
            token.lemma_ for token in doc 
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
        ]
    else:
        # 简单分词：仅保留长度>2的字母组合，不处理停用词和词形还原
        words = [w for w in re.findall(r'\b[a-zA-Z]+\b', text.lower()) if len(w) > 2]
        
    bigrams = [" ".join(words[j:j+2]) for j in range(len(words)-1)]
    trigrams = [" ".join(words[j:j+3]) for j in range(len(words)-2)]
    return words, bigrams, trigrams

def build_custom_vocab(dataset_name=DATA_NAME, sample_size=SAMPLE_SIZE):
    print(f"开始从 {dataset_name} 加载流式数据...")
    
    # 2. 加载流式数据集
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    
    print(f"正在处理前 {sample_size} 篇文章...")
    
    # 3. 处理数据
    print(f"使用 {MAX_WORKERS} 个线程并行处理...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_text, example['text']) for example in dataset.take(sample_size)]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=sample_size):
            try:
                words, bigrams, trigrams = future.result()
                
                unigram_counts.update(words)
                bigram_counts.update(bigrams)
                trigram_counts.update(trigrams)
            except Exception as e:
                print(f"Error processing text: {e}")

    # 4. 转换为 DataFrame
    df_unigrams = pd.DataFrame(unigram_counts.most_common(TOP_UNIGRAMS), columns=['Word', 'Frequency'])
    df_bigrams = pd.DataFrame(bigram_counts.most_common(TOP_BIGRAMS), columns=['Phrase', 'Frequency'])
    df_trigrams = pd.DataFrame(trigram_counts.most_common(TOP_TRIGRAMS), columns=['Trigram', 'Frequency'])
    
    # Filter by MIN_FREQUENCY
    df_bigrams = df_bigrams[df_bigrams['Frequency'] > MIN_FREQUENCY]
    df_trigrams = df_trigrams[df_trigrams['Frequency'] > MIN_FREQUENCY]
    
    # 5. Dump 到 CSV
    df_unigrams.to_csv("words.csv", index=False)
    df_bigrams.to_csv("phrases.csv", index=False)
    df_trigrams.to_csv("trigrams.csv", index=False)
    
    print("任务完成！词表已保存至: words.csv, phrases.csv 和 trigrams.csv")

if __name__ == "__main__":
    # sample_size 越大，词频越准，但运行时间越长
    import time
    start = time.time()
    build_custom_vocab()
    end = time.time()
    print(f"运行时间: {end - start} 秒")
