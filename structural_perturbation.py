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

    # Step 2: Compute Delta_M
    V = Vt.T
    M_transpose_M_P = M_P.T @ M_P
    M_transpose_M = M.T @ M
    Delta_M = M_transpose_M - M_transpose_M_P
    if timing_flag:
        print(f"Step 2 (Delta_M computation): {time.time() - step_time:.4f} seconds")
        step_time = time.time()

    # Step 3: Compute Delta_Sigma_T_Sigma and Delta_Sigma
    Delta_Sigma_T_Sigma = np.diag(Vt @ Delta_M @ Vt.T)
    Sigma_sq = Sigma.diagonal() ** 2
    Delta_Sigma = np.sqrt(Sigma_sq + Delta_Sigma_T_Sigma) - Sigma.diagonal()
    if timing_flag:
        print(f"Step 3 (Delta_Sigma computation): {time.time() - step_time:.4f} seconds")
        step_time = time.time()

    # Step 4: Compute Sigma_tilde and M_tilde
    Sigma_tilde = Sigma + diags(Delta_Sigma)
    Sigma_tilde_diag = Sigma_tilde.diagonal()
    Sigma_tilde_Vt = Sigma_tilde_diag[:, np.newaxis] * Vt

    # Compute M_tilde in chunks to save memory
    chunk_size = 10000
    M_tilde = np.zeros((U.shape[0], Sigma_tilde_Vt.shape[1]))

    for i in range(0, U.shape[0], chunk_size):
        M_tilde[i:i+chunk_size, :] = np.dot(U[i:i+chunk_size, :], Sigma_tilde_Vt)

    if timing_flag:
        print(f"Step 4 (M_tilde computation): {time.time() - step_time:.4f} seconds")

    return M_tilde, Sigma, Sigma_tilde


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

    Example:
    --------
    >>> import pandas as pd
    >>> # Load your data
    >>> df = pd.DataFrame({
    ...     'user_id': [1, 1, 2, 2, 3],
    ...     'item_id': [1, 2, 1, 3, 2],
    ...     'rating': [5, 4, 3, 5, 4],
    ...     'timestamp': pd.date_range('2020-01-01', periods=5)
    ... })
    >>> rmse, std, s_dist, std_s, rmse_svd, std_svd = \
    ...     analytical_structural_perturbation_v2(df, p=0.2, n_iterations=5)
    >>> print(f"Structural Perturbation RMSE: {rmse:.4f} ± {std:.4f}")
    """
    np.random.seed(42)

    df_random = train_df.copy()
    M, user_map, item_map = convert_to_sparse_matrix(train_df)
    M = M.astype('float32')

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
        M_tilde, Sigma, Sigma_tilde = compute_perturbation_impact(M, M_P, n_components,
                                                                   timing_flag=False)

        # Combine all perturbed indices
        if R_M_nonzero.size == 0:
            permuted_indices = original_permuted_indices
        else:
            permuted_indices = np.concatenate((original_permuted_indices,
                                              R_M_zero, R_M_nonzero), axis=0)

        # Calculate RMSE for analytical approach
        true_ratings = np.array([M[row, col] for row, col in permuted_indices])
        predicted_ratings = np.array([M_tilde[row, col] for row, col in permuted_indices])
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
        Sigma_M_diag = diags(Sigma_M)
        Sigma_M_Vt = Sigma_M_diag @ Vt_M

        M_svd = np.zeros((U_M.shape[0], Sigma_M_Vt.shape[1]))
        chunk_size = 10000
        for j in range(0, U_M.shape[0], chunk_size):
            M_svd[j:j+chunk_size, :] = np.dot(U_M[j:j+chunk_size, :], Sigma_M_Vt)

        # Calculate RMSE for standard SVD
        true_ratings = np.array([M[row, col] for row, col in permuted_indices])
        predicted_ratings = np.array([M_svd[row, col] for row, col in permuted_indices])
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
