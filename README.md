# UGD-LTR: Uncertainty-Aware Graph-Based Deep Learning Framework for Learning to Rank

## Overview

UGD-LTR (Uncertainty-aware Graph-based Deep Learning to Rank) is a novel framework that addresses two fundamental limitations in learning-to-rank systems:
1. **Neglecting query-document dependencies** - Traditional methods treat query-document pairs as isolated instances
2. **Missing uncertainty quantification** - Most approaches don't account for prediction disagreement among rankers

The framework models ranking data as a weighted bipartite graph, extracts 22 structural features capturing centrality, neighborhood, and projection-based properties, then fuses multiple base rankers using the Choquet fuzzy integral for uncertainty-aware aggregation.

## Key Features

- **Graph-Based Feature Extraction**: Converts LTR datasets into bipartite graphs and extracts 22 structural features
- **Deep Feature Prediction**: Trains specialized neural networks to predict graph features for cold-start scenarios
- **Uncertainty-Aware Fusion**: Combines multiple rankers using Choquet fuzzy integrals to handle prediction disagreement
- **Competitive Performance**: Outperforms state-of-the-art methods including HLambdaMART, RankSVM-Struct, and SeaRank

## Performance Highlights

| Dataset | Metric | Improvement vs. Best Baseline |
|---------|--------|-------------------------------|
| WCL2R | NDCG@1 | +5.6% over SeaRank |
| WCL2R | P@1 | +11.7% over SeaRank |
| MQ2007 | P@1 | +1.8% over HLambdaMART |
| MQ2008 | P@1 | +3.2% over RankSVM-Struct |


## Datasets

The framework is evaluated on three benchmark datasets:

| Dataset | Queries | Documents | Features | Data Type |
|---------|---------|-----------|----------|-----------|
| MQ2007 (LETOR4.0) | 1,692 | 65,323 | 46 | Web search (GOV2) |
| MQ2008 (LETOR4.0) | 784 | 14,384 | 46 | Web search (GOV2) |
| WCL2R | 277,095 | 3.1M | 29 | Clickthrough + IR features |

### Download Instructions

- **LETOR4.0**: Available at https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/
- **WCL2R**: Available from the Journal of Information and Data Management (requires registration)

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone repository
git clone https://github.com/[your-username]/UGD-LTR.git
cd UGD-LTR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.10.0
networkx>=2.8.0
lightgbm>=4.0.0
xgboost>=1.7.0
matplotlib>=3.5.0
joblib>=1.1.0
scipy>=1.9.0
```

## Usage

### Google Colab (Recommended)

The main notebook `UGD_LTR_WCL2R.ipynb` is designed for Google Colab:

1. Upload the notebook to Google Colab
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Place datasets in `/content/drive/MyDrive/WCL2R/FS.txt`
4. Place evaluation script in `/content/drive/MyDrive/Eval-Score-4.0-WCL2R.pl`
5. Run cells sequentially

### Local Execution

```python
# Load dataset
df = processFile('path/to/FS.txt')

# Split train/test (70/20/10 for WCL2R)
df_train, df_test = train_test_split2(df, test_size=0.29, random_state=42)

# Extract graph-based features
df_train_with_features = create_bipartite_graph_with_all_features(df_train)

# Predict graph features for test set
df_test_with_features, models, scalers = predict_centrality_features(
    df_train_with_features, df_test, top_k_features_array
)

# Train base rankers and fuse via Choquet integral
# (See notebook for complete pipeline)
```

## Algorithm Pipeline

### Step 1: Graph-Based Feature Extraction

The framework constructs a weighted bipartite graph:
- **Nodes**: Queries and documents
- **Edges**: Relevance labels between queries and documents

Extracted features (22 total):

| Category | Features |
|----------|----------|
| Basic Node | Weighted Degree (Document/Query) |
| Centrality | PageRank, Betweenness, Closeness, Eigenvector |
| Neighborhood | Avg. Neighbor Degree, Clustering Coefficient, Common Neighbors Ratio |
| Projection | Document/Query Projection Degree & Clustering |

### Step 2: Deep Feature Prediction

Since test-time graph construction is impossible, specialized deep models predict graph features from standard features:
- Constructs Feature Similarity Graph using Kendall's Tau correlation
- Applies spectral clustering for feature selection
- Trains 22 independent feedforward networks (one per graph feature)

### Step 3: Fuzzy Fusion

Combines base rankers (KNN, Random Forest, LightGBM) using Choquet fuzzy integral:
- Learns fuzzy measure λ from validation performance
- Captures interactions and synergies between rankers
- Handles prediction disagreement through non-linear aggregation

## Configuration

Optimal hyperparameters (determined via grid search):

| Parameter | MQ2007 | MQ2008 | WCL2R |
|-----------|--------|--------|-------|
| Graph pruning threshold (σ) | 0.12 | 0.12 | 0.12 |
| # Clusters (spectral) | 4 | 3 | 4 |
| Top features (%) | 30 | 30 | 30 |
| Hidden layers | 3 | 3 | 4 |
| Dropout rates | [0.5,0.4,0.3] | [0.5,0.4,0.3] | [0.5,0.4,0.3,0.2] |
| Learning rate | 0.005 | 0.005 | 0.001 |
| Batch size | 32 | 32 | 32 |

## Evaluation Metrics

The framework evaluates using standard IR metrics:
- **P@n** (Precision at n): Accuracy within top-n positions
- **NDCG@n** (Normalized Discounted Cumulative Gain): Graded relevance with position discount
- **MAP** (Mean Average Precision): Average over precision-recall curve

Run evaluation:
```bash
perl Eval-Score-4.0-WCL2R.pl judgements.txt predictions.txt output.txt 0
```

## Results

### LETOR4-MQ2007

| Method | NDCG@1 | P@1 | MAP |
|--------|--------|-----|-----|
| HLambdaMART | 0.4685 | 0.5375 | 0.5026 |
| **UGD-LTR** | **0.4715** | **0.5472** | 0.4944 |

### LETOR4-MQ2008

| Method | NDCG@1 | P@1 | MAP |
|--------|--------|-----|-----|
| RankSVM-Struct | 0.4156 | 0.4963 | 0.5279 |
| **UGD-LTR** | **0.4402** | **0.5165** | 0.5160 |

### WCL2R

| Method | NDCG@1 | P@1 | MAP |
|--------|--------|-----|-----|
| SeaRank | 0.6548 | 0.5828 | 0.5922 |
| **UGD-LTR** | **0.6915** | **0.6508** | **0.6088** |

## Ablation Study

| Configuration | WCL2R NDCG@1 | Improvement |
|---------------|--------------|-------------|
| Standard only | 0.6421 | Baseline |
| Graph only | 0.6653 | +2.3% |
| + Average fusion | 0.6801 | +3.8% |
| **Full UGD-LTR** | **0.6915** | **+4.9%** |

## Computational Requirements

| Dataset | Training Time (GPU) | Inference Time (per query) |
|---------|--------------------|----------------------------|
| MQ2007 | ~28 min | ~3.4 ms |
| MQ2008 | ~22 min | ~3.1 ms |
| WCL2R | ~140 min | ~4.2 ms |

**Note**: The 22 deep predictors are embarrassingly parallel and can be trained simultaneously on multi-core systems (17.9× speedup with 22 cores).

## Limitations

- **Sparse graph regions**: Performance degrades for queries with few document connections
- **Static graphs**: Requires retraining for dynamic document collections
- **Top-rank focus**: Optimized for precision at top positions, may sacrifice recall
- **Distributional shifts**: Vulnerable to changes between training and deployment environments

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{piroozmand2025ugdltr,
  title={UGD-LTR: An Uncertainty-Aware Graph-Based Deep Learning Framework for Learning to Rank},
  author={Piroozmand, Maryam and Moeini, Ali and Mazoochi, Mojtaba and Keyhanipour, Amir Hosein},
  journal={Applied Intelligence},
  year={2025}
}
```

## Acknowledgments

This research was supported by the ICT Research Institute, Tehran, Iran.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Maryam Piroozmand: m.piroozmand@ut.ac.ir
- Ali Moeini (Corresponding Author): moeini@ut.ac.ir
