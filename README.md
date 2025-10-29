# Project: Text Readability Prediction with Multi-Layered Feature Engineering

This repository contains the final project for Natural Language Processing, implementing a machine learning pipeline to predict text readability scores from the CommonLit Readability Dataset.

## Project Overview

The goal of this project is to develop a high-accuracy readability prediction model. We employ a feature-driven approach that integrates three distinct sets of NLP features:

- **Statistical Features:** Capturing foundational text structure (e.g., avg. word/sentence length).
- **Lexical Features:** TF-IDF vectors (including unigrams and bigrams).
- **Semantic Features:** Bigram-aware embeddings generated using `gensim` and a pre-trained `GloVe` model.

These combined features are fed into a Random Forest Regressor to predict the final readability score.

## Final Results

Our methodology was validated through a three-phase experiment. Our final model, which combines all three feature sets, achieved a **Root Mean Squared Error (RMSE) of 0.6997** on a 20% held-out test set. This significantly outperformed our baseline model (Statistical Features only, RMSE: 0.8884).

A detailed analysis, discussion of pros/cons, and qualitative examples are available in our final report: [Report_A Feature-Based Approach to Predicting Text Readability.pdf](Report_A Feature-Based Approach to Predicting Text Readability.pdf).

## How to Run This Project

This project was developed using Python 3.12.

### 1. Data Download

This project requires the `train.csv` file from the CommonLit Readability Dataset.

- Please download `train.csv` from its source (e.g., the Kaggle competition page).
- **Important:** Place the `train.csv` file in the same root directory as the Jupyter Notebook.

### 2. Environment Setup

Using a virtual environment is strongly recommended to manage dependencies.

```bash
# 1. Clone this repository (if using Git)
git clone https://github.com/mingjie-wei/Project_Text_Readability_Prediction.git
cd Project_Text_Readability_Prediction

# 2. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install all required Python packages 
pip install -r requirements.txt
```

### 3. Running the Project
- Start the Jupyter Notebook / Lab server

- In your browser, open the modeling.ipynb file.

- First-Time Setup Notes (Inside the Notebook):
    - The notebook code will automatically download the required NLTK models (`punkt` and `punkt_tab`).
    - It will also download the `glove-wiki-gigaword-100` pre-trained model via `gensim`.
    - A stable internet connection is required for these first steps. The GloVe model download is ~130MB and may take a few minutes.

- You can run all cells in order ("Cell" -> "Run All") to reproduce the full pipeline, from data loading through all three phases of model training and final evaluation.

## File Structure
```
.
├── .gitignore
├── data/
│   └── train.csv          # The dataset (must be downloaded manually)
├── notebooks/
│   └── modeling.ipynb     # The main notebook with all code and analysis
├── Makefile               # (Optional: development commands)
├── README.md              # This file
├── requirements.txt       # Python package dependencies
└── report.pdf             # Final report
```