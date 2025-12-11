# Structural Complexity and Predictability in Recommender Systems

This repository implements methods to measure structural complexity in recommender systems using matrix factorization and perturbation theory.

## Quick Start

### Using the Standalone Module

The easiest way to analyze your data:

```bash
# Run the example with synthetic data
python example_usage.py

# Or run the built-in demo
python structural_perturbation.py
```

### Using Your Own Data

```python
from structural_perturbation import analytical_structural_perturbation_v2
import pandas as pd

# Load your data (must have columns: user_id, item_id, rating, timestamp)
df = pd.read_csv('your_ratings.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Run structural perturbation analysis
rmse, std, s_dist, std_s, rmse_svd, std_svd = \
    analytical_structural_perturbation_v2(
        df,
        p=0.1,              # Perturb 10% of ratings
        n_iterations=5,     # Average over 5 runs
        n_components=50,    # Number of latent factors
        alpha=0.7,          # 70% value perturbation, 30% structural
        time_sampling=True  # Weight newer ratings in sampling
    )

print(f"Structural Perturbation RMSE: {rmse:.4f} ± {std:.4f}")
print(f"Spectral Distance: {s_dist:.4f} ± {std_s:.4f}")
print(f"Normalized RMSE: {rmse/rmse_svd:.4f}")
```

## Files

- **structural_perturbation.py** - Main module with all analysis functions
- **example_usage.py** - Example script showing how to use the module
- **predictability.ipynb** - Original notebook implementation
- **predictability_results.ipynb** - Results analysis and visualization
- **CLAUDE.md** - Detailed architecture documentation for AI assistants

## Key Concepts

### Structural Perturbation Analysis

This method measures how sensitive a recommender system's latent structure is to perturbations in the rating matrix. It implements:

1. **Value Perturbation**: Randomly permute existing rating values
2. **Structural Perturbation**: Move ratings from non-zero to zero positions

### Parameters

- **p**: Fraction of ratings to perturb (default: 0.1)
- **alpha**: Balance between value (alpha) and structural (1-alpha) perturbations (default: 0.7)
- **n_components**: Number of latent factors in SVD (default: 100)
- **n_iterations**: Number of runs to average (default: 1)
- **time_sampling**: Whether to weight newer ratings more in sampling (default: True)

### Outputs

- **RMSE**: Root mean squared error on perturbed entries
- **Spectral Distance**: Relative error in singular values
- **Normalized RMSE**: RMSE divided by standard SVD baseline

Lower values indicate more structural consistency and predictability.

## Requirements

```bash
pip install numpy pandas scipy scikit-learn
```

For the full evaluation pipeline (notebooks):
```bash
pip install surprise matplotlib
# Note: rs_datasets is a custom dependency for loading standard datasets
```

## Citation

If you use this code in your research, please cite the original work.
