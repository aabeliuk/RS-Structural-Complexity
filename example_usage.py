"""
Example usage of structural_perturbation module.

This script demonstrates how to use the analytical structural perturbation
analysis on your own recommendation data.
"""

import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2


def load_your_data():
    """
    Replace this function with your own data loading logic.

    Your data should be a pandas DataFrame with columns:
    - user_id: User identifier (int or str)
    - item_id: Item identifier (int or str)
    - rating: Rating value (numeric)
    - timestamp: Timestamp of rating (datetime or numeric)

    Returns:
    --------
    pd.DataFrame with columns: user_id, item_id, rating, timestamp
    """
    # Example: Load from CSV
    # df = pd.read_csv('ratings.csv')
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    # return df

    # For demonstration, create synthetic data
    np.random.seed(42)
    n_ratings = 10000

    df = pd.DataFrame({
        'user_id': np.random.randint(0, 500, n_ratings),
        'item_id': np.random.randint(0, 200, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings),
        'timestamp': pd.date_range('2020-01-01', periods=n_ratings, freq='h')
    })

    return df


def main():
    # Load your data
    print("Loading data...")
    df = load_your_data()

    print(f"Loaded {len(df)} ratings")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Items: {df['item_id'].nunique()}")
    print(f"Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4f}")
    print()

    # Run structural perturbation analysis
    print("Running structural perturbation analysis...")
    print("This may take a few minutes depending on dataset size...")
    print()

    # Configure parameters
    p = 0.1              # Perturb 10% of ratings
    n_iterations = 5     # Average over 5 runs
    n_components = 50    # Use 50 latent factors
    alpha = 0.7          # 70% value perturbation, 30% structural perturbation
    time_sampling = True # Weight newer ratings more in sampling

    rmse, std_rmse, s_distance, std_s_distance, rmse_svd, std_rmse_svd = \
        analytical_structural_perturbation_v2(
            df,
            p=p,
            n_iterations=n_iterations,
            n_components=n_components,
            alpha=alpha,
            time_sampling=time_sampling
        )

    # Display results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Structural Perturbation RMSE: {rmse:.4f} ± {std_rmse:.4f}")
    print(f"Spectral Distance:            {s_distance:.4f} ± {std_s_distance:.4f}")
    print(f"Standard SVD RMSE:            {rmse_svd:.4f} ± {std_rmse_svd:.4f}")
    print(f"Normalized RMSE:              {rmse/rmse_svd:.4f}")
    print()
    print("Interpretation:")
    print(f"  - Lower RMSE indicates more structural consistency")
    print(f"  - Lower spectral distance indicates more stable latent structure")
    print(f"  - Normalized RMSE allows comparison across different datasets")
    print("=" * 60)


if __name__ == "__main__":
    main()
