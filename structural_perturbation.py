"""
Structural Perturbation Analysis for Recommender Systems

This module implements analytical structural perturbation methods to measure
the complexity and predictability of recommendation systems.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import time


def sample_zero_forever(mat):
    """
    Generator to yield random zero indices from a sparse matrix.

    Parameters:
    -----------
    mat : scipy.sparse matrix
        The sparse matrix to sample from

    Yields:
    -------
    tuple : (row, col) indices of zero entries
    """
    nonzero_or_sampled = set(zip(*mat.nonzero()))
    while True:
        t = (np.random.randint(0, mat.shape[0]), np.random.randint(0, mat.shape[1]))
        if t not in nonzero_or_sampled:
            yield t
            nonzero_or_sampled.add(t)


def convert_to_sparse_matrix(df):
    """
    Convert a DataFrame with user-item ratings to a sparse matrix.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: user_id, item_id, rating

    Returns:
    --------
    rating_matrix : scipy.sparse.csr_matrix
        Sparse matrix of ratings
    user_map : pd.Series
        Mapping from user_id to matrix row index
    item_map : pd.Series
        Mapping from item_id to matrix column index
    """
    # Ensure the DataFrame is sorted
    df = df.sort_values(['user_id', 'item_id'])

    # Aggregate ratings by taking the mean
    ratings = df.groupby(['user_id', 'item_id'], sort=False)['rating'].mean().reset_index()

    # Create mappings for users and items
    user_map = pd.Series(index=ratings['user_id'].unique(),
                         data=np.arange(ratings['user_id'].nunique()))
    item_map = pd.Series(index=ratings['item_id'].unique(),
                         data=np.arange(ratings['item_id'].nunique()))

    # Map user and item IDs to indices
    ratings['user_idx'] = user_map[ratings['user_id']].values
    ratings['item_idx'] = item_map[ratings['item_id']].values

    # Create the sparse rating matrix
    rating_matrix = csr_matrix((ratings['rating'],
                                (ratings['user_idx'], ratings['item_idx'])))

    return rating_matrix, user_map, item_map


def compute_perturbation_impact(M, M_P, n_components, timing_flag=False):
    """
    Compute the impact of perturbation on the SVD factorization.

    This function implements the analytical perturbation theory approach to
    estimate how the singular values and reconstructed matrix change under
    perturbation.

    Parameters:
    -----------
    M : scipy.sparse matrix
        Original rating matrix
    M_P : scipy.sparse matrix
        Perturbed rating matrix
    n_components : int
        Number of singular values to compute
    timing_flag : bool, optional
        Print timing information for each step

    Returns:
    --------
    M_tilde : np.ndarray
        Reconstructed matrix with corrected singular values
    Sigma : scipy.sparse.diags
        Original singular values (diagonal matrix)
    Sigma_tilde : scipy.sparse.diags
        Corrected singular values (diagonal matrix)
    """
    if timing_flag:
        start_time = time.time()

    # Step 1: Compute SVD of M_P
    U, Sigma, Vt = svds(M_P, k=n_components)
    Sigma = diags(Sigma)
    if timing_flag:
        print(f"Step 1 (SVD computation): {time.time() - start_time:.4f} seconds")
        step_time = time.time()

    # Step 2: Compute Delta_Sigma_T_Sigma efficiently (memory-optimized)
    # Avoid creating huge item×item matrices by computing only what we need
    # Mathematical equivalence: diag(Vt @ Delta_M @ Vt.T) = diag((M@Vt.T).T @ (M@Vt.T) - (M_P@Vt.T).T @ (M_P@Vt.T))

    M_Vt = M @ Vt.T      # Sparse @ dense -> dense (users × k, ~32MB)
    M_P_Vt = M_P @ Vt.T  # Sparse @ dense -> dense (users × k, ~32MB)

    # Compute k×k matrix (tiny!) instead of item×item matrix (huge!)
    Delta_M_projected = M_Vt.T @ M_Vt - M_P_Vt.T @ M_P_Vt  # k×k (50×50, ~20KB)

    if timing_flag:
        print(f"Step 2 (Delta_Sigma_T_Sigma computation): {time.time() - step_time:.4f} seconds")
        step_time = time.time()

    # Step 3: Compute Delta_Sigma from projected Delta_M
    Delta_Sigma_T_Sigma = np.diag(Delta_M_projected)
    Sigma_sq = Sigma.diagonal() ** 2
    Delta_Sigma = np.sqrt(Sigma_sq + Delta_Sigma_T_Sigma) - Sigma.diagonal()
    if timing_flag:
        print(f"Step 3 (Delta_Sigma computation): {time.time() - step_time:.4f} seconds")
        step_time = time.time()

    # Step 4: Prepare components for on-demand prediction (memory-optimized)
    Sigma_tilde = Sigma + diags(Delta_Sigma)

    if timing_flag:
        print(f"Step 4 (Sigma_tilde computation): {time.time() - step_time:.4f} seconds")

    # Return components instead of materialized M_tilde (saves 16.6GB!)
    # Caller can compute predictions as: U @ Sigma_tilde @ Vt
    return (U, Sigma_tilde.diagonal(), Vt), Sigma, Sigma_tilde


def analytical_structural_perturbation_v2(train_df, p=0.1, n_iterations=1,
                                          n_components=100, alpha=0.7,
                                          time_sampling=True):
    """
    Calculate structural perturbation metrics using analytical perturbation theory.

    This function measures how sensitive a recommender system's latent structure is
    to perturbations in the rating matrix. It implements both value perturbations
    (permuting existing ratings) and structural perturbations (moving ratings from
    non-zero to zero positions).

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with columns: user_id, item_id, rating, timestamp
    p : float, optional (default=0.1)
        Fraction of ratings to perturb
    n_iterations : int, optional (default=1)
        Number of iterations to average results
    n_components : int, optional (default=100)
        Number of latent factors (singular values) to use
    alpha : float, optional (default=0.7)
        Fraction of perturbations that are value-based (vs structural)
        - alpha=1.0: only permute rating values
        - alpha=0.0: only move ratings to zero positions
        Note: For binary/implicit feedback datasets (all ratings are 0 or 1),
        alpha is automatically set to 0 since value permutation has no effect
        when all values are identical. This ensures the full perturbation
        amount p is applied through structural perturbations only.
    time_sampling : bool, optional (default=True)
        If True, sample newer ratings with higher probability

    Returns:
    --------
    average_rmse : float
        Mean RMSE between true and predicted ratings on perturbed entries
    std_rmse : float
        Standard deviation of RMSE across iterations
    average_s_distance : float
        Mean spectral distance (relative error in singular values)
    std_s_distance : float
        Standard deviation of spectral distance
    average_rmse_svd : float
        Mean RMSE using standard SVD (for normalization)
    std_rmse_svd : float
        Standard deviation of SVD RMSE

    """
    np.random.seed(42)

    df_random = train_df.copy()
    M, user_map, item_map = convert_to_sparse_matrix(train_df)
    M = M.astype('float32')

    # Detect binary/implicit feedback data
    unique_values = np.unique(M.data)
    is_binary = len(unique_values) <= 2 and all(v in [0.0, 1.0] for v in unique_values)

    if is_binary and alpha > 0:
        print(f"  Binary/implicit feedback detected (unique values: {sorted(unique_values)})")
        print(f"  Adjusting alpha: {alpha} → 0.0 (structural perturbations only)")
        print(f"  Reason: Permuting identical values has no effect on binary data")
        alpha = 0.0

    # Adjust n_components if too large
    n_components = min(n_components, int(min(M.shape) / 4))
    total_known_cells = M.nnz
    num_selected_ratings = int(total_known_cells * p * alpha)

    rmse_values = []
    rmse_svd_values = []
    s_distance_values = []

    for i in range(n_iterations):
        # Prepare sampling probabilities based on timestamps
        timestamps = df_random['timestamp'].astype('int64') / 1e9
        normalized_timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-9)
        if time_sampling:
            probabilities = normalized_timestamps / normalized_timestamps.sum()
        else:
            probabilities = None

        # Sample and permute ratings (value perturbation)
        indices = np.random.choice(df_random.index, size=num_selected_ratings,
                                   replace=False, p=probabilities)
        df_random.loc[indices, 'rating'] = np.random.permutation(
            df_random.loc[indices, 'rating'].values)

        # Map selected dataframe indices to (row, col) in the matrix M
        original_permuted_indices = [
            (user_map[train_df.loc[idx, 'user_id']],
             item_map[train_df.loc[idx, 'item_id']])
            for idx in indices
        ]

        # Create perturbed matrix
        M_P, _, _ = convert_to_sparse_matrix(df_random)
        M_P = M_P.astype('float32')
        M_P = M_P.tolil()

        # Structural perturbation: move ratings from non-zero to zero positions
        nonzero_indices = np.array(list(zip(M.nonzero()[0], M.nonzero()[1])))
        num_selected_nonzero = int((1.0 - alpha) * p * len(nonzero_indices))
        selected_nonzero_indices = np.random.choice(nonzero_indices.shape[0],
                                                    num_selected_nonzero,
                                                    replace=False)
        R_M_nonzero = nonzero_indices[selected_nonzero_indices]

        itr = sample_zero_forever(M)
        R_M_zero = np.array([next(itr) for _ in range(num_selected_nonzero)])

        for idx_nonzero, idx_zero in zip(R_M_nonzero, R_M_zero):
            M_P[idx_zero[0], idx_zero[1]] = M[idx_nonzero[0], idx_nonzero[1]]
            M_P[idx_nonzero[0], idx_nonzero[1]] = 0

        M_P = M_P.tocsr()

        # Compute perturbation impact
        (U, sigma_tilde_diag, Vt), Sigma, Sigma_tilde = compute_perturbation_impact(M, M_P, n_components,
                                                                                      timing_flag=False)

        # Combine all perturbed indices
        if R_M_nonzero.size == 0:
            permuted_indices = original_permuted_indices
        else:
            permuted_indices = np.concatenate((original_permuted_indices,
                                              R_M_zero, R_M_nonzero), axis=0)

        # Calculate RMSE for analytical approach
        # Compute predictions on-demand instead of materializing full M_tilde
        true_ratings = np.array([M[row, col] for row, col in permuted_indices])
        predicted_ratings = np.array([
            np.dot(U[row, :] * sigma_tilde_diag, Vt[:, col])
            for row, col in permuted_indices
        ])
        rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        rmse_values.append(rmse)

        # Calculate spectral distance
        lambda_true = Sigma.diagonal()
        lambda_approx = Sigma_tilde.diagonal()
        relative_errors = np.abs(lambda_true - lambda_approx) / lambda_true
        s_distance = np.mean(relative_errors)
        s_distance_values.append(s_distance)

        # Perform standard SVD on M for normalization
        U_M, Sigma_M, Vt_M = svds(M, k=n_components)
        Sigma_M_diag = Sigma_M

        # Calculate RMSE for standard SVD
        # Compute predictions on-demand instead of materializing full M_svd
        true_ratings = np.array([M[row, col] for row, col in permuted_indices])
        predicted_ratings = np.array([
            np.dot(U_M[row, :] * Sigma_M_diag, Vt_M[:, col])
            for row, col in permuted_indices
        ])
        rmse_svd = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
        rmse_svd_values.append(rmse_svd)

    # Return averaged results
    average_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    average_s_distance = np.mean(s_distance_values)
    std_s_distance = np.std(s_distance_values)
    average_rmse_svd = np.mean(rmse_svd_values)
    std_rmse_svd = np.std(rmse_svd_values)

    return average_rmse, std_rmse, average_s_distance, std_s_distance, average_rmse_svd, std_rmse_svd


def main():
    """
    Example usage of the structural perturbation analysis.
    """
    # Create a simple example dataset
    np.random.seed(42)

    # Generate synthetic rating data
    n_users = 100
    n_items = 50
    n_ratings = 1000

    user_ids = np.random.randint(0, n_users, n_ratings)
    item_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)
    timestamps = pd.date_range('2020-01-01', periods=n_ratings, freq='h')

    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })

    print("Running structural perturbation analysis...")
    print(f"Dataset: {n_users} users, {n_items} items, {n_ratings} ratings")
    print()

    # Run analysis with different parameter settings
    settings = [
        {'p': 0.1, 'alpha': 1.0, 'time_sampling': True, 'n_components': 20},
        {'p': 0.1, 'alpha': 0.7, 'time_sampling': True, 'n_components': 20},
        {'p': 0.1, 'alpha': 0.5, 'time_sampling': False, 'n_components': 20},
    ]

    for i, params in enumerate(settings, 1):
        print(f"Configuration {i}: p={params['p']}, alpha={params['alpha']}, "
              f"time_sampling={params['time_sampling']}, n_components={params['n_components']}")

        rmse, std, s_dist, std_s, rmse_svd, std_svd = \
            analytical_structural_perturbation_v2(df, **params, n_iterations=3)

        print(f"  Structural Perturbation RMSE: {rmse:.4f} ± {std:.4f}")
        print(f"  Spectral Distance: {s_dist:.4f} ± {std_s:.4f}")
        print(f"  Standard SVD RMSE: {rmse_svd:.4f} ± {std_svd:.4f}")
        print(f"  Normalized RMSE: {rmse/rmse_svd:.4f}")
        print()


if __name__ == "__main__":
    main()
