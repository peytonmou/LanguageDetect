# Language Detection Comparison

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-green.svg)](https://spacy.io/)

## Overview
This project compares two language detection libraries - `langdetect` and `lingua` - using a custom multilingual dataset created by scraping children's stories from WorldStories.org.uk across English, Spanish, Italian, and Chinese.

## Features
- Web scraping using BeautifulSoup
- Sentence tokenization with spaCy
- Language detection using langdetect and lingua
- Performance evaluation with confusion matrices and F1-scores
- Testing on code-mixed sentences

## Dataset
- 3 children's books × 4 languages × 30 sentences = 360 sentences
- Additional 8 code-mixed sentences for limit testing

## Installation

```bash
git clone https://github.com/yourusername/language-detection-comparison.git
cd language-detection-comparison
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download it_core_news_sm
python -m spacy download zh_core_web_sm

## Requirements
requests==2.31.0
beautifulsoup4==4.12.2
pandas==2.0.3
spacy==3.7.2
langdetect==1.0.9
lingua-language-detector==1.3.2
scikit-learn==1.3.0
