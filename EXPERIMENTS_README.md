# Multi-Dataset RS Experiment Script - User Guide

## Overview

This script (`run_experiments.py`) runs comprehensive experiments comparing different sampling techniques across multiple datasets and recommendation system algorithms. It implements **Relative Performance Analysis (RPA)** to measure how different sampling strategies affect model performance compared to using 100% of the training data.

## Features

- **Multiple Datasets**: Test across multiple Amazon review datasets
- **Multiple Algorithms**: Compare LightGCN, BPR, and NeuMF
- **Sampling Strategies**:
  - **Difficult**: Train on the most difficult-to-predict ratings
  - **Random**: Random sampling (baseline)
  - **Easiest**: Train on the easiest-to-predict ratings
- **Sampling Rates**: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
- **Metrics**: Precision@10, NDCG@10, MAP@10
- **Relative Performance Analysis**: Calculate % improvement/loss vs 100% baseline
- **Comprehensive Visualizations**: Metric plots, RPA plots, and aggregated comparisons
- **Caching**: Automatically caches perturbation analysis results

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy torch recbole
```

## Quick Start

### 1. Basic Usage

Simply run the script:

```bash
python run_experiments.py
```

This will:
1. Load all configured datasets
2. Compute difficulty rankings for each dataset (cached for future runs)
3. Run experiments for all algorithms and sampling strategies
4. Save results to CSV files
5. Generate visualizations

### 2. Configuration

Edit the `CONFIG` dictionary at the top of `run_experiments.py`:

```python
CONFIG = {
    # Datasets to test
    'datasets': [
        'Amazon_Health_and_Personal_Care',
        'Amazon_Grocery_and_Gourmet_Food'
    ],

    # Algorithms to test
    'algorithms': ['LightGCN', 'BPR', 'NeuMF'],

    # Sampling rates (%)
    'sampling_rates': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

    # ... other parameters
}
```

### 3. Adding New Datasets

To add a new dataset:

1. Ensure the dataset is available in RecBole format in the `dataset/` directory
2. Add the dataset name to the `CONFIG['datasets']` list
3. Run the script

Example:
```python
CONFIG = {
    'datasets': [
        'Amazon_Health_and_Personal_Care',
        'Amazon_Grocery_and_Gourmet_Food',
        'Amazon_Books',  # Add new dataset
    ],
    # ...
}
```

## Output Structure

```
results/
├── Amazon_Health_and_Personal_Care_LightGCN.csv
├── Amazon_Health_and_Personal_Care_BPR.csv
├── Amazon_Health_and_Personal_Care_NeuMF.csv
├── Amazon_Grocery_and_Gourmet_Food_LightGCN.csv
├── Amazon_Grocery_and_Gourmet_Food_BPR.csv
├── Amazon_Grocery_and_Gourmet_Food_NeuMF.csv
└── all_results_summary.csv

plots/
├── Amazon_Health_and_Personal_Care_LightGCN_metrics.png
├── Amazon_Health_and_Personal_Care_LightGCN_rpa.png
├── Amazon_Health_and_Personal_Care_BPR_metrics.png
├── Amazon_Health_and_Personal_Care_BPR_rpa.png
├── ... (one metrics + one RPA plot per dataset-algorithm combination)
├── aggregated_rpa.png
└── heatmap_50pct_sampling.png

cache/
└── difficult_ratings_Amazon_Health_and_Personal_Care.pkl
└── difficult_ratings_Amazon_Grocery_and_Gourmet_Food.pkl
```

## Understanding the Results

### CSV Output Format

Each CSV file contains columns:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `algorithm` | RS algorithm (LightGCN, BPR, NeuMF) |
| `sampling_rate` | Percentage of training data used (10-100) |
| `strategy` | Sampling strategy (difficult, random, difficult_inverse) |
| `precision` | Precision@10 |
| `ndcg` | NDCG@10 |
| `map` | MAP@10 |
| `n_ratings` | Number of ratings in training sample |
| `precision_rpa` | Precision RPA (% vs 100% baseline) |
| `ndcg_rpa` | NDCG RPA (% vs 100% baseline) |
| `map_rpa` | MAP RPA (% vs 100% baseline) |

### Relative Performance Analysis (RPA)

RPA measures the percentage change in performance compared to using 100% of the data:

```
RPA = (metric@X% - metric@100%) / metric@100% * 100
```

**Interpretation:**
- **Negative values**: Performance loss (e.g., -20% means 20% worse than 100%)
- **Zero**: Same performance as 100% baseline
- **Positive values**: Performance gain (rare, indicates the sample is better than full data)

**Example:**
If NDCG@100% = 0.0500 and NDCG@50% = 0.0450:
- RPA = (0.0450 - 0.0500) / 0.0500 * 100 = -10%
- Interpretation: Using 50% of data results in 10% performance loss

### Visualizations

1. **Metric Plots** (`*_metrics.png`):
   - Shows absolute metric values across sampling rates
   - Compares all three strategies
   - One plot per dataset-algorithm combination

2. **RPA Plots** (`*_rpa.png`):
   - Shows relative performance loss/gain vs 100% baseline
   - Horizontal line at 0% represents the baseline
   - Helps identify which strategy loses less performance with less data

3. **Aggregated RPA Plot** (`aggregated_rpa.png`):
   - Average RPA across all datasets and algorithms
   - Shows general trends across different domains

4. **Heatmap** (`heatmap_50pct_sampling.png`):
   - Shows best NDCG@10 at 50% sampling
   - Compares performance across dataset × algorithm combinations

