import os
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from typing import List, Dict
import sys

# 加载环境变量
load_dotenv()

# 配置
INPUT_FILE = "words.csv"
OUTPUT_FILE = "words_explained.csv"
BATCH_SIZE = 50          # 每批处理单词数 (Batching to reduce overhead)
CONCURRENT_LIMIT = 32    # 最大并发请求数 (Semaphore to avoid Rate Limits)
MODEL_NAME = os.getenv("MODEL_NAME") or "deepseek-chat" # 使用性价比高的模型，或者 gpt-3.5-turbo
BASE_URL = os.getenv("BASE_URL") or ""
API_KEY = os.getenv("API_KEY") or ""

if not BASE_URL or not API_KEY:
    print("Error: BASE_URL or API_KEY not found in environment.")
    sys.exit(1)

async def translate_batch(client: AsyncOpenAI, words: List[str], semaphore: asyncio.Semaphore) -> Dict[str, str]:
    """
    异步翻译一批单词，返回 {word: translation} 字典
    """
    async with semaphore:
        prompt = (
            f"Please translate the following English words into concise Simplified Chinese meanings. "
            f"Output a valid JSON object where the key is the English word and the value is the Chinese translation. "
            f"Words: {json.dumps(words)}"
        )

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise English-Chinese translator. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"Error translating batch: {e}")
            return {}

async def main():
    # 1. 初始化客户端
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # 2. 读取数据
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        # 确保有 Word 列
        if "Word" not in df.columns:
            raise ValueError("CSV must contain a 'Word' column")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 提取单词列表 (处理 NaN)
    all_words = df["Word"].dropna().astype(str).tolist()
    total_words = len(all_words)
    print(f"Found {total_words} words. Starting translation...")

    # 3. 准备异步任务
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    tasks = []
    
    # 分批创建任务
    for i in range(0, total_words, BATCH_SIZE):
        batch = all_words[i : i + BATCH_SIZE]
        tasks.append(translate_batch(client, batch, semaphore))

    # 4. 执行并收集结果
    results = await tqdm.gather(*tasks, desc="Translating Batches")

    # 5. 合并结果
    translation_map = {}
    for res in results:
        translation_map.update(res)

    # 6. 将结果映射回 DataFrame
    # 使用 map 填充，未找到的填入 None 或 原文
    df["Translation"] = df["Word"].map(translation_map)

    # 7. 保存结果
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())

