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
SKIP_ROWS = int(os.getenv("SKIP_ROWS", "0"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "50000"))

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

def save_vocab(unigram_counts, bigram_counts, trigram_counts):
    # 转换为 DataFrame
    df_unigrams = pd.DataFrame(unigram_counts.most_common(TOP_UNIGRAMS), columns=['Word', 'Frequency'])
    df_bigrams = pd.DataFrame(bigram_counts.most_common(TOP_BIGRAMS), columns=['Phrase', 'Frequency'])
    df_trigrams = pd.DataFrame(trigram_counts.most_common(TOP_TRIGRAMS), columns=['Trigram', 'Frequency'])
    
    # Filter by MIN_FREQUENCY
    df_bigrams = df_bigrams[df_bigrams['Frequency'] > MIN_FREQUENCY]
    df_trigrams = df_trigrams[df_trigrams['Frequency'] > MIN_FREQUENCY]
    
    # Dump 到 CSV
    df_unigrams.to_csv("words.csv", index=False)
    df_bigrams.to_csv("phrases.csv", index=False)
    df_trigrams.to_csv("trigrams.csv", index=False)
    print(f"\n[Checkpoint] 词表已保存 (Unigrams: {len(df_unigrams)}, Bigrams: {len(df_bigrams)}, Trigrams: {len(df_trigrams)})")

def build_custom_vocab(dataset_name=DATA_NAME, sample_size=SAMPLE_SIZE):
    print(f"开始从 {dataset_name} 加载流式数据...")
    
    # 2. 加载流式数据集
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    if SKIP_ROWS > 0:
        print(f"跳过前 {SKIP_ROWS} 篇文章...")
        dataset = dataset.skip(SKIP_ROWS)

    # 如果 sample_size 为负数或极大的数，则尽可能多地处理（受限于数据集大小）
    iterable_dataset = dataset
    if sample_size > 0:
        iterable_dataset = dataset.take(sample_size)
    
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    
    target_desc = f"前 {sample_size}" if sample_size > 0 else "所有可用"
    print(f"正在处理{target_desc}篇文章...")
    
    # 3. 处理数据
    print(f"使用 {MAX_WORKERS} 个线程并行处理...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 set 来管理正在运行的任务，控制内存占用
        futures = set()
        # 适当的 buffer 大小，保证 worker 不闲置即可，太大会占用内存
        MAX_PENDING_FUTURES = MAX_WORKERS * 2 
        
        # 创建进度条
        pbar = tqdm(total=sample_size if sample_size > 0 else None, desc="Processing")
        processed_count = 0
        
        for example in iterable_dataset:
            processed_count += 1
            # 如果积压的任务太多，就等待完成一些
            if len(futures) >= MAX_PENDING_FUTURES:
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                # 处理已完成的任务
                for future in done:
                    try:
                        words, bigrams, trigrams = future.result()
                        unigram_counts.update(words)
                        bigram_counts.update(bigrams)
                        trigram_counts.update(trigrams)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing text: {e}")
                        pbar.update(1) # 即使出错也要更新进度

                # 定期保存 Checkpoint
                # 注意：这里使用的是已提交任务的近似计数，实际完成的可能略少，但足够用于定期保存
                if processed_count % CHECKPOINT_INTERVAL == 0:
                    save_vocab(unigram_counts, bigram_counts, trigram_counts)
            
            # 提交新任务
            futures.add(executor.submit(process_text, example['text']))
            
        # 处理剩余的任务
        for future in concurrent.futures.as_completed(futures):
            try:
                words, bigrams, trigrams = future.result()
                unigram_counts.update(words)
                bigram_counts.update(bigrams)
                trigram_counts.update(trigrams)
                pbar.update(1)
            except Exception as e:
                print(f"Error processing text: {e}")
                pbar.update(1)
                
        pbar.close()
        
        if sample_size > 0 and processed_count < sample_size:
            print(f"\n注意: 数据集只有 {processed_count} 篇文章，少于请求的 {sample_size} 篇。已处理所有可用数据。")
        else:
            print(f"\n成功处理了 {processed_count} 篇文章。")

    # 4. 最终保存
    save_vocab(unigram_counts, bigram_counts, trigram_counts)
    print("任务完成！")

if __name__ == "__main__":
    # sample_size 越大，词频越准，但运行时间越长
    import time
    start = time.time()
    build_custom_vocab()
    end = time.time()
    print(f"运行时间: {end - start} 秒")
