# Text Summarization with Transformers

This project builds an AI system that summarizes long articles into short, readable summaries using abstractive summarization. It fine-tunes a BART transformer model on the XSum dataset and deploys a browser-based tool using Streamlit.

## Features

- Fine-tuned BART model for abstractive summarization
- Evaluation with ROUGE and BLEU scores
- Web-based interface for easy summarization

## Setup

1. Clone or download the project.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install transformers datasets evaluate torch streamlit rouge-score`

## Usage

### Training
Run `python train.py` to fine-tune the model. This may take time depending on your hardware.

### Evaluation
Run `python evaluate.py` to compute ROUGE and BLEU scores on the test set.

### Web App
Run `streamlit run app.py` to start the web app. Paste text and get summaries.

## Files

- `data_prep.py`: Loads and preprocesses the XSum dataset
- `train.py`: Fine-tunes the BART model
- `evaluate.py`: Evaluates the model
- `app.py`: Streamlit web app
- `TODO.md`: Project tasks

## Dataset

Uses the CNN/DailyMail dataset from Hugging Face, which contains news articles and multi-sentence summaries.

## Model

BART-base fine-tuned for summarization.
