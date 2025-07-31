# DarijaLLM

A Language Model for Moroccan Darija (الدارجة المغربية)

This project is a personal learning initiative focused on training Large Language Models (LLMs) specifically for Moroccan Darija, a dialect of Arabic spoken in Morocco. The project includes data collection, preprocessing, tokenization, and model training components.

## Project Overview

DarijaLLM aims to create a language model that can understand and generate text in Moroccan Darija. The project follows a complete machine learning pipeline from data collection to model training, including web scraping, data cleaning, custom tokenization, and model pretraining.

## Project Structure

### 📁 Root Directory Files

- **`environment.yml`** - Conda environment configuration file containing all required Python packages and dependencies for the project
- **`README.md`** - This documentation file explaining the project structure and components

### 📁 Jupyter Notebooks

- **`WebScrapper.ipynb`** - Web scraping script that collects Moroccan Darija articles from goud.ma website
  - Scrapes articles by date from 2020 to present
  - Extracts title, date, content link, and article content
  - Saves raw data to CSV format

- **`DataProcessing.ipynb`** - Data preprocessing and cleaning pipeline
  - Cleans and normalizes the scraped text data
  - Removes HTML tags, special characters, and formatting
  - Prepares data for tokenization and model training

- **`Tokenizer.ipynb`** - Custom tokenizer training using Byte Pair Encoding (BPE)
  - Implements BPE algorithm for Darija text
  - Trains tokenizer on cleaned Darija corpus
  - Saves trained tokenizer model

- **`Pretraining.ipynb`** - Main model training notebook
  - Uses HuggingFace transformers library
  - Trains a language model on the processed Darija data
  - Implements training with wandb logging for experiment tracking

### 📁 Utils Directory

- **`utils/__init__.py`** - Python package initialization file
- **`utils/BPE.py`** - Custom Byte Pair Encoding implementation
  - Complete BPE algorithm implementation from scratch
  - Includes training, encoding, and model saving/loading functionality
  - Supports custom vocabulary size and merge operations

### 📁 Assets Directory

#### Data Files (`assets/data/`)
- **`articles.csv`** - Raw scraped articles from web scraping
- **`articles_cleaned.csv`** - Processed and cleaned article data
- **`articles_tokenized.csv`** - Tokenized version of the articles for model training

#### Language Identification Data (`assets/data-LID/`)
- **`train_Darija_LID.csv`** - Training dataset for language identification
- **`test_Darija_LID.csv`** - Test dataset for language identification
- **`data_train.txt`** - Text format training data
- **`data_test.txt`** - Text format test data
- **`data_val.txt`** - Validation dataset

#### Models (`assets/models/`)
- **`BPE.pkl`** - Saved Byte Pair Encoding model with learned merges
- **`LID.ftz`** - Language Identification model (1.7GB) for detecting Darija text

#### Other Assets
- **`darijaStopWord.csv`** - List of stop words in Moroccan Darija
- **`NotoNaskhArabic-VariableFont_wght.ttf`** - Arabic font file for text rendering

## Key Features

### Data Collection
- Automated web scraping from Moroccan news websites
- Date-based article collection spanning multiple years
- Robust error handling and progress tracking

### Text Processing
- Arabic text normalization and cleaning
- Diacritics removal and Unicode normalization
- Custom stop word filtering for Darija

### Tokenization
- Custom BPE implementation specifically designed for Arabic/Darija
- Handles Arabic script characteristics
- Configurable vocabulary size and merge operations

### Model Training
- Uses state-of-the-art transformer architecture
- Implements proper train/test splits
- Includes experiment tracking with wandb
- Supports both custom and pretrained tokenizers

## Technical Stack

- **Python 3.10** - Main programming language
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace library for transformer models
- **Pandas** - Data manipulation and analysis
- **BeautifulSoup** - Web scraping
- **Scikit-learn** - Machine learning utilities
- **Wandb** - Experiment tracking
- **Jupyter Notebooks** - Interactive development environment

## Getting Started

1. **Environment Setup**
   ```bash
   conda env create -f environment.yml
   conda activate LLMTest
   ```

2. **Data Collection**
   - Run `WebScrapper.ipynb` to collect articles
   - This will create the initial `articles.csv` file

3. **Data Processing**
   - Run `DataProcessing.ipynb` to clean and prepare the data
   - This creates `articles_cleaned.csv`

4. **Tokenization**
   - Run `Tokenizer.ipynb` to train the custom BPE tokenizer
   - This saves the tokenizer model to `assets/models/BPE.pkl`

5. **Model Training**
   - Run `Pretraining.ipynb` to train the language model
   - Monitor training progress with wandb

## Project Goals

- Create a functional language model for Moroccan Darija
- Demonstrate complete ML pipeline from data to model
- Learn and implement modern NLP techniques
- Contribute to Arabic dialect processing research

## Notes

This is a personal learning project focused on understanding the complete process of training language models. The project demonstrates various NLP techniques including web scraping, text preprocessing, custom tokenization, and transformer model training.

