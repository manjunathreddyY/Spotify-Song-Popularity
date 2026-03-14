# Spotify Song Popularity Prediction

A machine learning project that predicts Spotify track popularity scores using audio features and metadata. Multiple regression models are trained and compared, from a simple baseline through to gradient-boosted ensembles.

## Project Overview

Spotify assigns each track a popularity score (0–100) based on streaming metrics. This project explores whether audio characteristics like danceability, energy, tempo, and loudness — along with metadata such as genre and artist — can predict that score.

The analysis covers the full ML pipeline: exploratory data analysis, feature engineering, class imbalance handling (SMOTE), model training with cross-validation, hyperparameter tuning, and model comparison.

## Dataset

The dataset contains **32,833 tracks** sourced from Spotify playlists, with 23 features per track.

**Key features used for modeling:**

- **Audio features:** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, key, mode
- **Metadata:** track_artist, playlist_genre, playlist_subgenre
- **Target variable:** track_popularity (0–100)

## Models Implemented

| Model | Description |
|-------|-------------|
| Baseline | Mean artist popularity |
| Linear Regression | With StandardScaler, 10-fold CV, and feature selection |
| Decision Tree | DecisionTreeRegressor with feature importance analysis |
| Random Forest | Ensemble of decision trees with cross-validation |
| XGBoost | Gradient boosting with GridSearchCV hyperparameter tuning |
| LightGBM | Light gradient boosting with GridSearchCV tuning |

Each model is evaluated on RMSE, MSE, and MAE, with improvement measured against the baseline.

## Project Structure

```
Spotify-Song-Popularity-main/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   └── spotify_songs.csv
└── notebooks/
    └── spotify_song_popularity_analysis.ipynb
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Spotify-Song-Popularity.git
   cd Spotify-Song-Popularity
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook notebooks/spotify_song_popularity_analysis.ipynb
   ```

## Methodology

1. **Exploratory Data Analysis** — Distribution plots, correlation heatmaps, and genre-level analysis to understand the data.
2. **Feature Engineering** — Label encoding of categorical variables, Box-Cox transformations, and feature selection using SelectKBest.
3. **Handling Class Imbalance** — SMOTE oversampling applied to the training set to address skewed popularity distributions.
4. **Model Training & Evaluation** — Each model is trained with K-Fold cross-validation and evaluated on a held-out test set using RMSE, MSE, and MAE.
5. **Hyperparameter Tuning** — GridSearchCV used for XGBoost and LightGBM to optimize performance.
6. **Model Comparison** — Side-by-side comparison of all models with improvement percentages over baseline.


