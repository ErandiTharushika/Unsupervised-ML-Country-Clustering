# Unsupervised-ML-Country-Clustering

Unsupervised learning on socio-economic country data — dimensionality reduction (PCA, t-SNE) and clustering (K-Means, DBSCAN) to segment countries by development level.

## Overview

This project applies unsupervised learning techniques to a dataset of 167 countries described by nine socio-economic indicators (child mortality, exports, health spending, imports, income, inflation, life expectancy, fertility rate, and GDP per capita). The goal is to reduce the dimensionality of the data, uncover natural groupings of countries, and interpret those groupings in a real-world context (e.g. "Developed," "Emerging," "Low-Income").

## Repository Structure

```
Unsupervised-ML-Country-Clustering/
├── notebooks/
│   ├── dimensionality_reduction_countrydata.ipynb   # PCA & t-SNE
│   └── clustering_countrydata.ipynb                 # K-Means & DBSCAN
├── dataset/
│   └── Country-data.csv
└── README.md
```

## Dataset

[Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) (Kaggle) — 167 countries, 9 numeric features, no missing values.

## Notebooks

### 1. `dimensionality_reduction_countrydata.ipynb`
- Data loading, inspection, and preprocessing (`StandardScaler`)
- **PCA**: covariance matrix, eigenvalues/eigenvectors, explained variance, 2D projection, top contributing features per component
- **t-SNE**: embeddings compared across a grid of perplexity / learning rate / iteration settings, with discussion of neighbourhood preservation

### 2. `clustering_countrydata.ipynb`
- **K-Means**: optimal *k* via elbow method and silhouette score, cluster visualization in PCA and t-SNE space, centroid interpretation
- **DBSCAN**: `eps` / `min_samples` tuning, noise/outlier detection and visualization
- Comparison of K-Means vs. DBSCAN (cluster shape, stability, interpretability)
- Final cluster labeling ("Developed Nations," "Emerging Economies," "Low-Income Nations") and key differentiating features

## Key Findings

- The first 2 principal components explain ~63% of total variance; ~5 components are needed to reach 90%.
- `gdpp` (GDP per capita) and `child_mort` (child mortality) are the strongest cluster differentiators.
- K-Means (k=3) cleanly separates countries into Developed, Emerging, and Low-Income groups.
- DBSCAN flags a subset of countries as outliers/noise — both very small, wealthy nations and countries with unusual indicator combinations.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`

## Getting Started

```bash
git clone https://github.com/ErandiTharushika/Unsupervised-ML-Country-Clustering.git
cd Unsupervised-ML-Country-Clustering
pip install -r requirements.txt
jupyter notebook notebooks/
```

