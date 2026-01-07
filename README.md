# Fine Words: AI-Powered Language Learning Vocabulary Builder
# Fine Words: AI 驱动的语言学习词汇构建器

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🇬🇧 English Introduction

**Fine Words** is a powerful tool designed to help language learners master foreign languages by analyzing real-world usage frequencies. It fetches massive text datasets from **Hugging Face**, extracts high-frequency words and phrases (n-grams), and uses **Large Language Models (LLMs)** to provide concise, context-aware translations and explanations.

### Key Features
- **Data Streaming**: Efficiently streams large datasets from Hugging Face without downloading the entire dataset.
- **N-gram Extraction**: Generates lists for **Unigrams** (single words), **Bigrams** (2-word phrases), and **Trigrams** (3-word phrases).
- **Dual Processing Modes**:
  - **CPU Mode**: Fast, multi-threaded processing using lightweight models (`en_core_web_sm`).
  - **GPU Mode**: High-accuracy processing using Transformer-based models (`en_core_web_trf`) with CUDA acceleration.
- **LLM Integration**: Automatically translates and explains the extracted vocabulary using OpenAI-compatible APIs (e.g., DeepSeek, GPT-4, etc.).
- **Customizable**: Configurable sample sizes, frequency thresholds, and models via environment variables.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fine-words.git
   cd fine-words
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy models**:
   - For CPU mode:
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - For GPU mode (requires CUDA):
     ```bash
     pip install spacy-transformers
     python -m spacy download en_core_web_trf
     ```

### Configuration

Create a `.env` file in the root directory (copy from `.env.example` if available) and configure your settings:

```ini
# Hugging Face Settings
DATA_NAME=wikitext          # Dataset name (e.g., wikitext, c4)
HF_ENDPOINT=https://hf-mirror.com

# Processing Settings
TOP_UNIGRAMS=1000           # Number of top words to save
TOP_BIGRAMS=100             # Number of top 2-word phrases
TOP_TRIGRAMS=100            # Number of top 3-word phrases
SAMPLE_SIZE=10000           # Number of documents to process
MAX_WORKERS=8               # CPU threads
BATCH_SIZE=200              # GPU batch size

# LLM Settings
OPENAI_API_KEY=sk-......
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### Usage

#### Step 1: Generate Frequency Lists

**Option A: CPU Mode (Fast & Lightweight)**
Best for quick analysis on standard hardware.
```bash
python hf-words-cpu.py
```

**Option B: GPU Mode (High Accuracy)**
Best for deep linguistic analysis using Transformers (requires NVIDIA GPU).
```bash
python hf-words-gpu.py
```

*Output: `words.csv`, `phrases.csv`, `trigrams.csv`*

#### Step 2: AI Explanation & Translation

Use an LLM to translate and explain the generated word list.
```bash
python llm-explain.py
```

*Output: `words_explained.csv` containing the original words and their AI-generated translations.*

---

<a name="chinese"></a>
## 🇨🇳 中文介绍

**Fine Words** 是一个旨在通过分析真实语境词频来辅助语言学习的强大工具。它从 **Hugging Face** 获取海量文本数据集，提取高频单词和短语（N-grams），并利用 **大语言模型 (LLMs)** 提供简洁、准确的中文翻译和解释。

### 主要功能
- **流式数据处理**: 高效流式传输 Hugging Face 大型数据集，无需下载完整数据。
- **多级词组提取**: 生成 **单词 (Unigrams)**、**双词短语 (Bigrams)** 和 **三词短语 (Trigrams)** 列表。
- **双重处理模式**:
  - **CPU 模式**: 使用轻量级模型 (`en_core_web_sm`) 进行快速、多线程处理。
  - **GPU 模式**: 使用基于 Transformer 的模型 (`en_core_web_trf`) 和 CUDA 加速，实现高精度分析。
- **LLM 集成**: 调用 OpenAI 兼容接口（如 DeepSeek, GPT-4 等）自动翻译和解释提取的词汇。
- **高度可配置**: 可通过环境变量自定义样本大小、频率阈值和使用的模型。

### 安装指南

1. **克隆仓库**:
   ```bash
   git clone https://github.com/yourusername/fine-words.git
   cd fine-words
   ```

2. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

3. **下载 spaCy 模型**:
   - CPU 模式:
     ```bash
     python -m spacy download en_core_web_sm
     ```
   - GPU 模式 (需要 CUDA):
     ```bash
     pip install spacy-transformers
     python -m spacy download en_core_web_trf
     ```

### 配置说明

在项目根目录下创建 `.env` 文件，并配置以下参数：

```ini
# Hugging Face 设置
DATA_NAME=wikitext          # 数据集名称 (如 wikitext, c4)
HF_ENDPOINT=https://hf-mirror.com

# 处理设置
TOP_UNIGRAMS=1000           # 保存的高频词数量
TOP_BIGRAMS=100             # 保存的高频双词短语数量
TOP_TRIGRAMS=100            # 保存的高频三词短语数量
SAMPLE_SIZE=10000           # 处理的文档样本数量
MAX_WORKERS=8               # CPU 线程数
BATCH_SIZE=200              # GPU 批处理大小

# LLM 设置
OPENAI_API_KEY=sk-......    # API 密钥
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat    # 模型名称
```

### 使用方法

#### 第一步：生成词频表

**选项 A: CPU 模式 (快速 & 轻量)**
适合在普通硬件上快速分析。
```bash
python hf-words-cpu.py
```

**选项 B: GPU 模式 (高精度)**
适合使用 Transformer 模型进行深度语言分析（需要 NVIDIA GPU）。
```bash
python hf-words-gpu.py
```

*生成文件: `words.csv`, `phrases.csv`, `trigrams.csv`*

#### 第二步：AI 翻译与解释

使用 LLM 为生成的单词表提供翻译和解释。
```bash
python llm-explain.py
```

*生成文件: `words_explained.csv`，包含原始单词及其 AI 生成的翻译。*

