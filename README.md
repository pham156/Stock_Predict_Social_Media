# Stock Market Analysis with Social Media Integration

CS 577 Project: Analyzing stock market data and social media sentiment for predictive modeling.

## Project Structure

Stock-Market-Analysis/
├── data_pipeline.py
├── eda.ipynb
├── organizer.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
└── data/
    ├── sp500_historical.csv
    └── sp500_historical_clean.csv

## Quick Start

1. Setup Environment
conda create -n stock-analysis python=3.11
conda activate stock-analysis
pip install -r requirements.txt

2. Data Collection
python data_pipeline.py

3. Exploratory Analysis
jupyter notebook eda.ipynb

## Current Features

- S&P 500 Data Pipeline
- Exploratory Data Analysis
- Data Preprocessing
- GPU Support

## Planned Features

- Social media sentiment analysis
- RAG pipeline
- LSTM/Transformer models
- Multi-modal data fusion

CS 577 - Machine Learning Project