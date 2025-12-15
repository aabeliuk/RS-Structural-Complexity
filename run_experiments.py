"""
Multi-Dataset RS Experiment Script with Relative Performance Analysis

This script runs comprehensive experiments comparing different sampling techniques
(difficult, random, inverted-difficult) across multiple datasets and RS algorithms.
It computes Relative Performance Analysis (RPA) to measure performance loss/gain
compared to 100% sampling baseline.

Author: Generated with Claude Code
Date: 2025-12-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import os
import pickle
import warnings
from datetime import datetime
import torch

# Import RecBole libraries
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, get_model, get_trainer
from recbole.data.interaction import Interaction
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader

# Import structural perturbation functions
from structural_perturbation import (
    convert_to_sparse_matrix,
    compute_perturbation_impact
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Datasets to process
    'datasets': [
        'Amazon_Health_and_Personal_Care',  # Test with just one dataset first
        # 'Amazon_Grocery_and_Gourmet_Food',
        # 'book-crossing',
        # 'lastfm',
        # 'ModCloth',
        # 'pinterest',
        # 'RateBeer',
        # 'steam',
        # 'yelp2022'
    ],

    # RS Algorithms to test
    # 'algorithms': ['LightGCN', 'BPR', 'NeuMF'],
    'algorithms': ['BPR'],

    # Sampling rates to test (%)
    # 'sampling_rates': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'sampling_rates': [10, 100],  # Test with just 10% and 100%

    # Sampling strategies
    'strategies': ['difficult', 'random', 'difficult_inverse', 'temporal'],

    # Experiment parameters
    'min_ratings': 10,        # Minimum ratings per user/item
    'min_total_ratings_per_user': 3,  # Minimum total interactions per user (for stratified sampling)
    'max_users': 100000,     # Maximum users to keep
    'max_items': 100000,     # Maximum items to keep
    'n_partitions': 10,      # Partitions for perturbation analysis
    'n_components': 50,     # SVD components for perturbation
    'eval_k': 10,            # Top-K for evaluation metrics
    'random_seed': 42,       # Random seed for reproducibility

    # RecBole training parameters
    'epochs': 50,
    'train_batch_size': 2048,
    'eval_batch_size': 4096,
    'learning_rate': 0.001,
    'embedding_size': 64,

    # Output directories
    'results_dir': 'results/',
    'plots_dir': 'plots/',
    'cache_dir': 'cache/',
}


# =============================================================================
# HELPER FUNCTIONS (from notebook)
# =============================================================================

def setup_directories():
    """Create output directories if they don't exist."""
    for dir_path in [CONFIG['results_dir'], CONFIG['plots_dir'], CONFIG['cache_dir']]:
        os.makedirs(dir_path, exist_ok=True)
    print("Output directories created/verified.")


def preprocess_data(df, min_r=10, max_n=100000, min_total_ratings=3):
    """
    Preprocess dataset by filtering and subsampling users and items.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset with columns: user_id, item_id, rating, timestamp
    min_r : int
        Minimum number of ratings per user/item
    max_n : int
        Maximum number of users/items to keep
    min_total_ratings : int
        Minimum total interactions per user (for stratified per-user sampling)

    Returns:
    --------
    pd.DataFrame : Filtered dataset
    """
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Sort by user_id and timestamp
    sort_cols = ['user_id']
    if 'timestamp' in df.columns:
        sort_cols.append('timestamp')
    df = df.sort_values(by=sort_cols)

    # Filter users with minimum ratings (ensure compatibility with stratified sampling)
    user_ratings_count = df['user_id'].value_counts()
    min_required = max(min_r, min_total_ratings)
    valid_users = user_ratings_count[user_ratings_count >= min_required].index
    df_filtered = df[df['user_id'].isin(valid_users)]

    # Filter items with minimum ratings
    item_ratings_count = df_filtered['item_id'].value_counts()
    valid_items = item_ratings_count[item_ratings_count >= min_r].index
    df_filtered = df_filtered[df_filtered['item_id'].isin(valid_items)]

    # Sample users if too many
    n_users = len(valid_users)
    if n_users > max_n:
        sampled_users = np.random.choice(valid_users, size=max_n, replace=False)
        df_filtered = df_filtered[df_filtered['user_id'].isin(sampled_users)]

    # Sample items if too many
    n_items = len(valid_items)
    if n_items > max_n:
        sampled_items = np.random.choice(valid_items, size=max_n, replace=False)
        df_filtered = df_filtered[df_filtered['item_id'].isin(sampled_items)]

    return df_filtered


def save_recbole_format(df, dataset_name, output_path='dataset/'):
    """
    Save DataFrame to RecBole's .inter format.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with columns: user_id, item_id, rating, timestamp
    dataset_name : str
        Name of the dataset
    output_path : str
        Output directory path

    Returns:
    --------
    str : Path to the dataset directory
    """
    dataset_dir = os.path.join(output_path, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    inter_file = os.path.join(dataset_dir, f'{dataset_name}.inter')

    df_export = df.copy()
    if 'timestamp' in df_export.columns and pd.api.types.is_datetime64_any_dtype(df_export['timestamp']):
        df_export['timestamp'] = df_export['timestamp'].astype('int64') // 10**9

    # FIXED: Don't convert tokens to integers - keep original tokens
    # This ensures token consistency between train and test data
    # RecBole can handle string tokens directly

    with open(inter_file, 'w') as f:
        header_cols = ['user_id:token', 'item_id:token', 'rating:float']
        if 'timestamp' in df_export.columns:
            header_cols.append('timestamp:float')
        f.write('\t'.join(header_cols) + '\n')

    cols_to_write = ['user_id', 'item_id', 'rating']
    if 'timestamp' in df_export.columns:
        cols_to_write.append('timestamp')

    df_export[cols_to_write].to_csv(inter_file, sep='\t', mode='a', header=False, index=False)
    return dataset_dir


def temporal_holdout_split(df, test_ratio=None, leave_n_out=1):
    """
    Split data into train and test sets based on temporal holdout.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with user_id, item_id, rating, timestamp
    test_ratio : float, optional
        Ratio of test data per user
    leave_n_out : int, optional
        Number of last interactions per user for test

    Returns:
    --------
    tuple : (train_df, test_df)
    """
    has_timestamp = 'timestamp' in df.columns

    if not has_timestamp:
        print("  Warning: No timestamp column found. Using sequential order for split.")
        df = df.copy()
        df['timestamp'] = range(len(df))

    if test_ratio:
        cutoff_indices = df.groupby('user_id').cumcount() / df.groupby('user_id')['user_id'].transform('count')
        train_df = df[cutoff_indices < (1 - test_ratio)]
        test_df = df[cutoff_indices >= (1 - test_ratio)]
    elif leave_n_out:
        cutoff_indices = df.groupby('user_id').cumcount()
        train_df = df[cutoff_indices < (df.groupby('user_id')['user_id'].transform('count') - leave_n_out)]
        test_df = df[cutoff_indices >= (df.groupby('user_id')['user_id'].transform('count') - leave_n_out)]
    else:
        raise ValueError("Either test_ratio or leave_n_out must be provided.")

    if not has_timestamp:
        train_df = train_df.drop(columns=['timestamp'])
        test_df = test_df.drop(columns=['timestamp'])

    return train_df, test_df





def build_test_dataloader_from_df(test_df, dataset, config):
    """
    Convert a test pandas DF to RecBole Interaction format
    and wrap it into a FullSortEvalDataLoader.
    """

    # Map raw tokens → dataset's internal numeric IDs
    uid_map = dataset.field2id_token['user_id']
    iid_map = dataset.field2id_token['item_id']

    # FIXED: Don't filter again - test_df is already filtered in run_single_experiment
    # Just map the tokens to IDs
    df = test_df.copy()

    # Handle both dict and array-like mappings
    if isinstance(uid_map, dict):
        df['user_id'] = df['user_id'].map(uid_map)
        df['item_id'] = df['item_id'].map(iid_map)
    else:
        # If it's an array/Series, create proper mapping
        uid_to_id = {token: idx for idx, token in enumerate(uid_map) if token is not None}
        iid_to_id = {token: idx for idx, token in enumerate(iid_map) if token is not None}

        df['user_id'] = df['user_id'].map(uid_to_id)
        df['item_id'] = df['item_id'].map(iid_to_id)

    # Drop any rows with unmappable tokens (NaN values)
    df = df.dropna(subset=['user_id', 'item_id'])

    # Convert to integer type (RecBole expects int, not float)
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)

    if len(df) == 0:
        raise ValueError(
            f"No test samples remain after token mapping! "
            f"Input had {len(test_df)} samples. "
            f"All tokens failed to map - this suggests a data format issue."
        )

    # Create RecBole Interaction object with dataset context
    # Create interaction dict with proper field names
    interaction_dict = {
        config['USER_ID_FIELD']: torch.tensor(df['user_id'].values, dtype=torch.long),
        config['ITEM_ID_FIELD']: torch.tensor(df['item_id'].values, dtype=torch.long)
    }

    interaction = Interaction(interaction_dict)

    # Manually set required attributes for FullSortEvalDataLoader
    # used_ids tracks which items each user has interacted with (from training data)
    interaction.used_ids = {}

    # Build FullSortEvalDataLoader
    test_loader = FullSortEvalDataLoader(
        config,
        dataset,
        interaction,
        shuffle=False
    )

    return test_loader


def train_model_with_fixed_test(train_df, global_test_df, model_type='BPR', config_dict=None):
    """
    Train a RecBole model using `train_df` and evaluate on `global_test_df`.
    Uses RecBole's standard data preparation with manual split ratios.

    Returns:
        model, config, trainer, dataset, test_data_loader
    """

    # -----------------------------------------------------------
    # 2.1 Save BOTH train and test to RecBole format
    # -----------------------------------------------------------
    tmp_dataset = "temp_train"

    # Combine train and test for RecBole
    combined_df = pd.concat([train_df, global_test_df], ignore_index=True)
    save_recbole_format(combined_df, tmp_dataset)

    n_train = len(train_df)
    n_test = len(global_test_df)

    # Calculate number of items to adjust topk dynamically
    n_items = train_df['item_id'].nunique()
    # Adjust topk to avoid "index k out of range" errors
    max_k = max(1, n_items - 1)  # At least 1, at most n_items-1
    topk_values = [k for k in [5, 10, 20] if k <= max_k]
    if not topk_values:
        topk_values = [max_k]  # Use max available if all standard values are too large

    # -----------------------------------------------------------
    # 2.2 Base RecBole configuration
    # -----------------------------------------------------------
    base_config = {
        'model': model_type,
        'dataset': tmp_dataset,
        'data_path': 'dataset/',

        # Fields
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},

        # Use exact split ratios for train/test
        'eval_args': {
            'split': {'RS': [n_train, 0, n_test]},
            'mode': 'full',
            'order': 'RO'  # Respect order (train first, then test)
        },

        # Performance settings
        'epochs': 20,
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'learning_rate': 0.001,
        'embedding_size': 64,

        'metrics': ['Recall', 'Precision', 'NDCG', 'Hit', 'MRR', 'MAP'],
        'topk': topk_values,  # Dynamically adjusted based on n_items

        'seed': 42,
        'reproducibility': True,

        'save_dataset': False,
        'save_dataloaders': False,
        'show_progress': False,
    }

    if config_dict:
        base_config.update(config_dict)

    # -----------------------------------------------------------
    # 2.3 Create dataset and data loaders
    # -----------------------------------------------------------
    config = Config(model=model_type, dataset=tmp_dataset, config_dict=base_config)
    dataset = create_dataset(config)

    # Use RecBole's standard data preparation
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # -----------------------------------------------------------
    # 2.4 Build model + trainer
    # -----------------------------------------------------------
    model_class = get_model(model_type)
    model = model_class(config, train_data.dataset).to(config['device'])

    trainer_class = get_trainer(config['MODEL_TYPE'], model_type)
    trainer = trainer_class(config, model)

    # Train on training data only (no validation)
    trainer.fit(train_data, valid_data=None, saved=False, show_progress=False)

    return model, config, trainer, dataset, test_data

def evaluate_model(model, config, trainer, test_data_loader, k=10):
    """
    Evaluate RecBole model on the provided test data loader.

    Parameters:
    -----------
    model : RecBole model
    config : Config
    trainer : Trainer
    test_data_loader : DataLoader
    k : int
        Top-K for evaluation

    Returns:
    --------
    tuple : (precision, ndcg, map_at_k)
    """

    result = trainer.evaluate(
        test_data_loader,
        load_best_model=False,
        show_progress=False
    )

    if not result:
        return 0.0, 0.0, 0.0

    # Use the largest available k if requested k is not available
    available_k = k
    if f'precision@{k}' not in result:
        # Find the largest k that was actually computed
        topk_values = config['topk']
        available_k = max([k_val for k_val in topk_values if k_val <= k], default=topk_values[-1] if topk_values else 1)

    precision = result.get(f'precision@{available_k}', 0.0)
    ndcg = result.get(f'ndcg@{available_k}', 0.0)
    map_k = result.get(f'map@{available_k}', 0.0)

    return precision, ndcg, map_k



# =============================================================================
# PERTURBATION ANALYSIS FUNCTIONS
# =============================================================================

def compute_difficult_ratings(df, dataset_name, force_recompute=False):
    """
    Compute perturbation impact to identify difficult-to-predict ratings.

    This function performs structural perturbation analysis to identify which
    ratings are most difficult to predict. Results are cached to avoid recomputation.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with user_id, item_id, rating, timestamp
    dataset_name : str
        Name of the dataset (for caching)
    force_recompute : bool
        Force recomputation even if cache exists

    Returns:
    --------
    dict : {
        'indices_sorted': np.array of indices sorted by difficulty (hardest first),
        'indices_sorted_inverted': np.array of indices sorted by difficulty (easiest first),
        'squared_errors': dict of {index: squared_error},
        'global_train_df': pd.DataFrame,
        'global_test_df': pd.DataFrame,
        'user_map': mapping from user_id to matrix row index,
        'item_map': mapping from item_id to matrix column index
    }
    """
    cache_file = os.path.join(CONFIG['cache_dir'], f'difficult_ratings_{dataset_name}_v3_per_user_temporal.pkl')

    # Check cache
    if not force_recompute and os.path.exists(cache_file):
        print(f"  Loading cached difficult ratings for {dataset_name}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"  Computing difficult ratings for {dataset_name}...")
    print(f"    This may take several minutes...")

    # Split into train/test using leave-one-out
    train_df, test_df = temporal_holdout_split(df, leave_n_out=1)

    # Use ALL remaining data for training (no 90/10 split)
    global_train_df = train_df
    global_test_df = test_df

    print(f"    Training set: {len(global_train_df):,} ratings")
    print(f"    Test set (leave-1-out): {len(global_test_df):,} ratings")

    # Create partitions
    all_indices = global_train_df.index.to_numpy()
    shuffled_indices = np.random.permutation(all_indices)
    index_partitions = np.array_split(shuffled_indices, CONFIG['n_partitions'])

    # Convert to sparse matrix
    df_random = global_train_df.copy()
    M, user_map, item_map = convert_to_sparse_matrix(global_train_df)
    M = M.astype('float32')

    # Adjust n_components if needed
    n_components = min(CONFIG['n_components'], int(min(M.shape) / 4))
    print(f"    Using {n_components} SVD components")

    # Compute perturbation impact
    squared_errors = {}

    for partition_idx, indices in enumerate(index_partitions, 1):
        print(f"    Processing partition {partition_idx}/{CONFIG['n_partitions']}...", end=" ")

        # Permute ratings in this partition
        df_random.loc[indices, 'rating'] = np.random.permutation(
            df_random.loc[indices, 'rating'].values
        )

        # Map dataframe indices to matrix coordinates
        mapped_indices = [
            (idx, user_map[global_train_df.loc[idx, 'user_id']],
             item_map[global_train_df.loc[idx, 'item_id']])
            for idx in indices
        ]

        # Create perturbed matrix
        M_P, _, _ = convert_to_sparse_matrix(df_random)
        M_P = M_P.astype('float32')

        # Compute perturbation impact
        M_tilde, Sigma, Sigma_tilde = compute_perturbation_impact(
            M, M_P, n_components, timing_flag=False
        )

        # Calculate squared error for each rating
        for idx, row, col in mapped_indices:
            true_rating = M[row, col]
            predicted_rating = M_tilde[row, col]
            squared_error = (true_rating - predicted_rating) ** 2
            squared_errors[idx] = squared_error

        avg_error = np.mean([squared_errors[idx] for idx, _, _ in mapped_indices])
        print(f"Done (avg error: {avg_error:.4f})")

    # Sort by squared error
    indices_sorted = np.array(sorted(squared_errors.keys(),
                                    key=lambda x: squared_errors[x], reverse=True))
    indices_sorted_inverted = np.array(sorted(squared_errors.keys(),
                                             key=lambda x: squared_errors[x], reverse=False))

    errors_sorted = np.array([squared_errors[idx] for idx in indices_sorted])

    print(f"    Difficulty statistics:")
    print(f"      Mean error: {errors_sorted.mean():.4f}")
    print(f"      Max error: {errors_sorted.max():.4f}")
    print(f"      Min error: {errors_sorted.min():.4f}")

    # Organize difficulty scores by user (for stratified per-user sampling)
    print(f"    Organizing per-user difficulty rankings...")
    per_user_difficulty = {}

    # Group once instead of filtering repeatedly (much faster!)
    grouped = global_train_df.groupby('user_id', sort=False)

    for user_id, user_data in grouped:
        # Get this user's rating indices (already filtered by groupby)
        user_indices = user_data.index.to_numpy()

        # Extract their difficulty scores
        user_errors = {
            idx: squared_errors[idx]
            for idx in user_indices
            if idx in squared_errors
        }

        # Sort by difficulty (descending = hardest first)
        user_indices_sorted = np.array(sorted(
            user_errors.keys(),
            key=lambda x: user_errors[x],
            reverse=True
        ))

        # Inverted (ascending = easiest first)
        user_indices_sorted_inverted = user_indices_sorted[::-1]

        # Sort by timestamp (descending = most recent first) for temporal strategy
        # user_data is already grouped, so we can sort it directly
        if 'timestamp' in user_data.columns:
            user_data_temporal = user_data.sort_values('timestamp', ascending=False)
        else:
            # Fallback: use index order (assuming higher indices = more recent)
            user_data_temporal = user_data.sort_index(ascending=False)

        user_indices_sorted_temporal = user_data_temporal.index.to_numpy()

        per_user_difficulty[user_id] = {
            'indices_sorted': user_indices_sorted,
            'indices_sorted_inverted': user_indices_sorted_inverted,
            'indices_sorted_temporal': user_indices_sorted_temporal,
            'squared_errors': user_errors,
            'n_ratings': len(user_indices)
        }

    print(f"    Organized {len(per_user_difficulty)} users")

    # Package results
    result = {
        'per_user_difficulty': per_user_difficulty,  # NEW: Per-user rankings
        'global_train_df': global_train_df,
        'global_test_df': global_test_df,
        'user_map': user_map,
        'item_map': item_map
    }

    # Cache results
    print(f"    Caching results to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result


# =============================================================================
# EXPERIMENT EXECUTION FUNCTIONS
# =============================================================================

def run_single_experiment(train_sample_df, global_test_df, algorithm, strategy,
                         sampling_rate, n_ratings):
    """
    Run a single experiment: train model and evaluate.

    Parameters:
    -----------
    train_sample_df : pd.DataFrame
        Training sample
    global_test_df : pd.DataFrame
        Fixed global test set
    algorithm : str
        RS algorithm name
    strategy : str
        Sampling strategy name
    sampling_rate : int
        Sampling rate percentage
    n_ratings : int
        Number of ratings in sample

    Returns:
    --------
    dict : Result dictionary with metrics
    """
    try:
        # CRITICAL FIX: Filter test set to only include users/items in training sample
        # This prevents empty test sets when training on small samples
        train_users = set(train_sample_df['user_id'].unique())
        train_items = set(train_sample_df['item_id'].unique())

        filtered_test_df = global_test_df[
            global_test_df['user_id'].isin(train_users) &
            global_test_df['item_id'].isin(train_items)
        ]

        # Check if we have any test samples
        if len(filtered_test_df) == 0:
            print(f"      WARNING: No valid test samples after filtering")
            return {
                'sampling_rate': sampling_rate,
                'strategy': strategy,
                'precision': 0.0,
                'ndcg': 0.0,
                'map': 0.0,
                'n_ratings': n_ratings,
                'status': 'no_test_data'
            }

        # Train model
        model, config, trainer, dataset, test_data = train_model_with_fixed_test(
            train_sample_df, filtered_test_df, model_type=algorithm
        )

        # Evaluate
        precision, ndcg, map_score = evaluate_model(
            model, config, trainer, test_data, k=CONFIG['eval_k']
        )

        # Clear GPU memory
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'sampling_rate': sampling_rate,
            'strategy': strategy,
            'precision': precision,
            'ndcg': ndcg,
            'map': map_score,
            'n_ratings': n_ratings,
            'status': 'success'
        }

    except Exception as e:
        import traceback
        print(f"      ERROR: {str(e)}")
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            'sampling_rate': sampling_rate,
            'strategy': strategy,
            'precision': 0.0,
            'ndcg': 0.0,
            'map': 0.0,
            'n_ratings': n_ratings,
            'status': 'failed',
            'error': str(e)
        }


def stratified_per_user_sample(global_train_df, per_user_difficulty,
                               sampling_rate, strategy='difficult'):
    """
    Sample ratings using stratified per-user sampling.

    Ensures:
    - Each user contributes X% of their ratings
    - Minimum 1 rating per user (even at low sampling rates)
    - Strategy (difficult/random/easy) applied per-user

    Parameters:
    -----------
    global_train_df : pd.DataFrame
        All available training data
    per_user_difficulty : dict
        Per-user difficulty rankings from compute_difficult_ratings()
        Structure: {user_id: {'indices_sorted': array, 'n_ratings': int, ...}}
    sampling_rate : int
        Percentage to sample (0-100)
    strategy : str
        'difficult', 'random', or 'difficult_inverse'

    Returns:
    --------
    pd.DataFrame : Sampled training data, shuffled
    """
    sampled_indices = []

    for user_id, user_data in per_user_difficulty.items():
        n_user_ratings = user_data['n_ratings']

        # Guarantee at least 1 rating per user
        n_samples = max(1, int(n_user_ratings * sampling_rate / 100))

        # Get indices based on strategy
        if strategy == 'difficult':
            # Hardest ratings first
            available_indices = user_data['indices_sorted']
        elif strategy == 'difficult_inverse':
            # Easiest ratings first
            available_indices = user_data['indices_sorted_inverted']
        elif strategy == 'random':
            # Random order (shuffle the sorted indices)
            available_indices = np.random.permutation(user_data['indices_sorted'])
        elif strategy == 'temporal':
            # Most recent ratings first (pre-computed)
            available_indices = user_data['indices_sorted_temporal']
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Handle edge case: user has no difficulty scores computed
        if len(available_indices) == 0:
            # Fallback: randomly sample from user's actual ratings
            user_mask = global_train_df['user_id'] == user_id
            user_all_indices = global_train_df[user_mask].index.to_numpy()
            available_indices = np.random.permutation(user_all_indices)

        # Take top n_samples indices for this user (limit to available)
        n_samples = min(n_samples, len(available_indices))
        selected = available_indices[:n_samples]
        sampled_indices.extend(selected)

    # Validate: check that all sampled indices exist in global_train_df (use set for O(1) lookups)
    valid_indices_set = set(global_train_df.index)
    sampled_indices = [idx for idx in sampled_indices if idx in valid_indices_set]

    if len(sampled_indices) == 0:
        raise ValueError("No valid indices after sampling!")

    # Create sampled dataframe and shuffle
    sampled_df = global_train_df.loc[sampled_indices].copy()
    sampled_df = sampled_df.sample(frac=1, random_state=CONFIG['random_seed'])

    return sampled_df


def run_experiments_for_dataset(dataset_name, algorithm, difficult_data):
    """
    Run all experiments for one dataset and one algorithm.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    algorithm : str
        RS algorithm name
    difficult_data : dict
        Perturbation analysis results from compute_difficult_ratings()

    Returns:
    --------
    pd.DataFrame : Results for all experiments
    """
    results = []

    # Extract data structures (updated for per-user sampling)
    global_train_df = difficult_data['global_train_df']
    global_test_df = difficult_data['global_test_df']
    per_user_difficulty = difficult_data['per_user_difficulty']

    for sampling_rate in CONFIG['sampling_rates']:
        # Calculate expected total samples (for reporting)
        total_samples = sum(
            max(1, int(user_data['n_ratings'] * sampling_rate / 100))
            for user_data in per_user_difficulty.values()
        )

        print(f"    Sampling rate: {sampling_rate}% (~{total_samples:,} ratings)")

        # Strategy 1: DIFFICULT (per-user)
        print(f"      [1/4] Per-user difficult sampling...", end=" ")
        difficult_df = stratified_per_user_sample(
            global_train_df, per_user_difficulty,
            sampling_rate, strategy='difficult'
        )

        result = run_single_experiment(
            difficult_df, global_test_df, algorithm,
            'difficult', sampling_rate, len(difficult_df)
        )
        result['dataset'] = dataset_name
        result['algorithm'] = algorithm
        results.append(result)
        print(f"P@{CONFIG['eval_k']}={result['precision']:.4f}")

        # Strategy 2: RANDOM (per-user)
        print(f"      [2/4] Per-user random sampling...", end=" ")
        random_df = stratified_per_user_sample(
            global_train_df, per_user_difficulty,
            sampling_rate, strategy='random'
        )

        result = run_single_experiment(
            random_df, global_test_df, algorithm,
            'random', sampling_rate, len(random_df)
        )
        result['dataset'] = dataset_name
        result['algorithm'] = algorithm
        results.append(result)
        print(f"P@{CONFIG['eval_k']}={result['precision']:.4f}")

        # Strategy 3: EASY (per-user)
        print(f"      [3/4] Per-user easiest sampling...", end=" ")
        easy_df = stratified_per_user_sample(
            global_train_df, per_user_difficulty,
            sampling_rate, strategy='difficult_inverse'
        )

        result = run_single_experiment(
            easy_df, global_test_df, algorithm,
            'difficult_inverse', sampling_rate, len(easy_df)
        )
        result['dataset'] = dataset_name
        result['algorithm'] = algorithm
        results.append(result)
        print(f"P@{CONFIG['eval_k']}={result['precision']:.4f}")

        # Strategy 4: TEMPORAL (per-user)
        print(f"      [4/4] Per-user temporal sampling...", end=" ")
        temporal_df = stratified_per_user_sample(
            global_train_df, per_user_difficulty,
            sampling_rate, strategy='temporal'
        )

        result = run_single_experiment(
            temporal_df, global_test_df, algorithm,
            'temporal', sampling_rate, len(temporal_df)
        )
        result['dataset'] = dataset_name
        result['algorithm'] = algorithm
        results.append(result)
        print(f"P@{CONFIG['eval_k']}={result['precision']:.4f}")

    return pd.DataFrame(results)


# =============================================================================
# RELATIVE PERFORMANCE ANALYSIS (RPA)
# =============================================================================

def calculate_rpa(results_df):
    """
    Calculate Relative Performance Analysis: percentage improvement vs 100% baseline.

    Formula: RPA = (metric@X% - metric@100%) / metric@100% * 100

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with columns: dataset, algorithm, sampling_rate, strategy, metrics

    Returns:
    --------
    pd.DataFrame : Results with additional RPA columns
    """
    results_df = results_df.copy()

    # Add RPA columns
    results_df['precision_rpa'] = 0.0
    results_df['ndcg_rpa'] = 0.0
    results_df['map_rpa'] = 0.0

    # Calculate RPA for each dataset + algorithm + strategy combination
    for (dataset, algorithm, strategy), group in results_df.groupby(['dataset', 'algorithm', 'strategy']):
        # Get 100% baseline
        baseline = group[group['sampling_rate'] == 100]
        if len(baseline) == 0:
            continue

        baseline_precision = baseline['precision'].values[0]
        baseline_ndcg = baseline['ndcg'].values[0]
        baseline_map = baseline['map'].values[0]

        # Avoid division by zero
        if baseline_precision == 0:
            baseline_precision = 1e-10
        if baseline_ndcg == 0:
            baseline_ndcg = 1e-10
        if baseline_map == 0:
            baseline_map = 1e-10

        # Calculate RPA for each sampling rate
        mask = ((results_df['dataset'] == dataset) &
                (results_df['algorithm'] == algorithm) &
                (results_df['strategy'] == strategy))

        results_df.loc[mask, 'precision_rpa'] = (
            (results_df.loc[mask, 'precision'] - baseline_precision) / baseline_precision * 100
        )
        results_df.loc[mask, 'ndcg_rpa'] = (
            (results_df.loc[mask, 'ndcg'] - baseline_ndcg) / baseline_ndcg * 100
        )
        results_df.loc[mask, 'map_rpa'] = (
            (results_df.loc[mask, 'map'] - baseline_map) / baseline_map * 100
        )

    return results_df


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_metrics(results_df, dataset_name, algorithm):
    """
    Generate standard metric plots (Precision, NDCG, MAP).

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results for this dataset and algorithm
    dataset_name : str
        Name of the dataset
    algorithm : str
        RS algorithm name
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color scheme
    color_difficult = '#E63946'
    color_random = '#457B9D'
    color_easy = '#1EC423'
    color_temporal = '#F77F00'

    # Filter data
    data = results_df[(results_df['dataset'] == dataset_name) &
                      (results_df['algorithm'] == algorithm)]

    difficult_data = data[data['strategy'] == 'difficult'].sort_values('sampling_rate')
    random_data = data[data['strategy'] == 'random'].sort_values('sampling_rate')
    easy_data = data[data['strategy'] == 'difficult_inverse'].sort_values('sampling_rate')
    temporal_data = data[data['strategy'] == 'temporal'].sort_values('sampling_rate')

    # Plot 1: Precision
    axes[0].plot(difficult_data['sampling_rate'], difficult_data['precision'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[0].plot(random_data['sampling_rate'], random_data['precision'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[0].plot(easy_data['sampling_rate'], easy_data['precision'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[0].plot(temporal_data['sampling_rate'], temporal_data['precision'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[0].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'Precision@{CONFIG["eval_k"]}', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Precision@{CONFIG["eval_k"]}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].set_xticks(CONFIG['sampling_rates'])

    # Plot 2: NDCG
    axes[1].plot(difficult_data['sampling_rate'], difficult_data['ndcg'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[1].plot(random_data['sampling_rate'], random_data['ndcg'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[1].plot(easy_data['sampling_rate'], easy_data['ndcg'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[1].plot(temporal_data['sampling_rate'], temporal_data['ndcg'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[1].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(f'NDCG@{CONFIG["eval_k"]}', fontsize=12, fontweight='bold')
    axes[1].set_title(f'NDCG@{CONFIG["eval_k"]}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].set_xticks(CONFIG['sampling_rates'])

    # Plot 3: MAP
    axes[2].plot(difficult_data['sampling_rate'], difficult_data['map'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[2].plot(random_data['sampling_rate'], random_data['map'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[2].plot(easy_data['sampling_rate'], easy_data['map'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[2].plot(temporal_data['sampling_rate'], temporal_data['map'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[2].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel(f'MAP@{CONFIG["eval_k"]}', fontsize=12, fontweight='bold')
    axes[2].set_title(f'MAP@{CONFIG["eval_k"]}', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=11, loc='best')
    axes[2].set_xticks(CONFIG['sampling_rates'])

    plt.suptitle(f'{dataset_name} - {algorithm}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    filename = f'{dataset_name}_{algorithm}_metrics.png'
    filepath = os.path.join(CONFIG['plots_dir'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved metrics plot: {filepath}")


def plot_rpa(results_df, dataset_name, algorithm):
    """
    Generate Relative Performance Analysis plots.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results with RPA columns
    dataset_name : str
        Name of the dataset
    algorithm : str
        RS algorithm name
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color scheme
    color_difficult = '#E63946'
    color_random = '#457B9D'
    color_easy = '#1EC423'
    color_temporal = '#F77F00'

    # Filter data (exclude 100% as it's the baseline)
    data = results_df[(results_df['dataset'] == dataset_name) &
                      (results_df['algorithm'] == algorithm) &
                      (results_df['sampling_rate'] != 100)]

    difficult_data = data[data['strategy'] == 'difficult'].sort_values('sampling_rate')
    random_data = data[data['strategy'] == 'random'].sort_values('sampling_rate')
    easy_data = data[data['strategy'] == 'difficult_inverse'].sort_values('sampling_rate')
    temporal_data = data[data['strategy'] == 'temporal'].sort_values('sampling_rate')

    # Plot 1: Precision RPA
    axes[0].plot(difficult_data['sampling_rate'], difficult_data['precision_rpa'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[0].plot(random_data['sampling_rate'], random_data['precision_rpa'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[0].plot(easy_data['sampling_rate'], easy_data['precision_rpa'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[0].plot(temporal_data['sampling_rate'], temporal_data['precision_rpa'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[0].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RPA (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Precision@{CONFIG["eval_k"]} RPA', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=11, loc='best')

    # Plot 2: NDCG RPA
    axes[1].plot(difficult_data['sampling_rate'], difficult_data['ndcg_rpa'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[1].plot(random_data['sampling_rate'], random_data['ndcg_rpa'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[1].plot(easy_data['sampling_rate'], easy_data['ndcg_rpa'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[1].plot(temporal_data['sampling_rate'], temporal_data['ndcg_rpa'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RPA (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'NDCG@{CONFIG["eval_k"]} RPA', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=11, loc='best')

    # Plot 3: MAP RPA
    axes[2].plot(difficult_data['sampling_rate'], difficult_data['map_rpa'],
                 marker='o', linewidth=2.5, markersize=8,
                 color=color_difficult, label='Difficult', linestyle='-')
    axes[2].plot(random_data['sampling_rate'], random_data['map_rpa'],
                 marker='s', linewidth=2.5, markersize=8,
                 color=color_random, label='Random', linestyle='--')
    axes[2].plot(easy_data['sampling_rate'], easy_data['map_rpa'],
                 marker='^', linewidth=2.5, markersize=8,
                 color=color_easy, label='Easiest', linestyle=':')
    axes[2].plot(temporal_data['sampling_rate'], temporal_data['map_rpa'],
                 marker='D', linewidth=2.5, markersize=8,
                 color=color_temporal, label='Temporal', linestyle='-.')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[2].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('RPA (%)', fontsize=12, fontweight='bold')
    axes[2].set_title(f'MAP@{CONFIG["eval_k"]} RPA', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=11, loc='best')

    plt.suptitle(f'{dataset_name} - {algorithm} - Relative Performance Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    filename = f'{dataset_name}_{algorithm}_rpa.png'
    filepath = os.path.join(CONFIG['plots_dir'], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"      Saved RPA plot: {filepath}")


def plot_aggregated_comparison(results_df):
    """
    Generate aggregated comparison plots across datasets (one plot per algorithm).

    Parameters:
    -----------
    results_df : pd.DataFrame
        All results with RPA columns
    """
    color_difficult = '#E63946'
    color_random = '#457B9D'
    color_easy = '#1EC423'

    # Get unique algorithms
    algorithms = results_df['algorithm'].unique()

    # Create one aggregated plot per algorithm
    for algorithm in algorithms:
        print(f"  Creating aggregated plot for {algorithm}...")

        # Filter data for this algorithm
        algo_data = results_df[results_df['algorithm'] == algorithm]

        # Plot 1: Average RPA across datasets for each strategy
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Exclude 100% sampling rate
        data = algo_data[algo_data['sampling_rate'] != 100]

        # Group by strategy and sampling rate, average across datasets only
        avg_rpa = data.groupby(['strategy', 'sampling_rate']).agg({
            'precision_rpa': 'mean',
            'ndcg_rpa': 'mean',
            'map_rpa': 'mean'
        }).reset_index()

        difficult_avg = avg_rpa[avg_rpa['strategy'] == 'difficult'].sort_values('sampling_rate')
        random_avg = avg_rpa[avg_rpa['strategy'] == 'random'].sort_values('sampling_rate')
        easy_avg = avg_rpa[avg_rpa['strategy'] == 'difficult_inverse'].sort_values('sampling_rate')

        # Precision RPA
        axes[0].plot(difficult_avg['sampling_rate'], difficult_avg['precision_rpa'],
                     marker='o', linewidth=2.5, markersize=8,
                     color=color_difficult, label='Difficult', linestyle='-')
        axes[0].plot(random_avg['sampling_rate'], random_avg['precision_rpa'],
                     marker='s', linewidth=2.5, markersize=8,
                     color=color_random, label='Random', linestyle='--')
        axes[0].plot(easy_avg['sampling_rate'], easy_avg['precision_rpa'],
                     marker='^', linewidth=2.5, markersize=8,
                     color=color_easy, label='Easiest', linestyle=':')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Average RPA (%)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Precision@{CONFIG["eval_k"]} - Average RPA', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=11, loc='best')

        # NDCG RPA
        axes[1].plot(difficult_avg['sampling_rate'], difficult_avg['ndcg_rpa'],
                     marker='o', linewidth=2.5, markersize=8,
                     color=color_difficult, label='Difficult', linestyle='-')
        axes[1].plot(random_avg['sampling_rate'], random_avg['ndcg_rpa'],
                     marker='s', linewidth=2.5, markersize=8,
                     color=color_random, label='Random', linestyle='--')
        axes[1].plot(easy_avg['sampling_rate'], easy_avg['ndcg_rpa'],
                     marker='^', linewidth=2.5, markersize=8,
                     color=color_easy, label='Easiest', linestyle=':')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average RPA (%)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'NDCG@{CONFIG["eval_k"]} - Average RPA', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=11, loc='best')

        # MAP RPA
        axes[2].plot(difficult_avg['sampling_rate'], difficult_avg['map_rpa'],
                     marker='o', linewidth=2.5, markersize=8,
                     color=color_difficult, label='Difficult', linestyle='-')
        axes[2].plot(random_avg['sampling_rate'], random_avg['map_rpa'],
                     marker='s', linewidth=2.5, markersize=8,
                     color=color_random, label='Random', linestyle='--')
        axes[2].plot(easy_avg['sampling_rate'], easy_avg['map_rpa'],
                     marker='^', linewidth=2.5, markersize=8,
                     color=color_easy, label='Easiest', linestyle=':')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[2].set_xlabel('Sampling Rate (%)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Average RPA (%)', fontsize=12, fontweight='bold')
        axes[2].set_title(f'MAP@{CONFIG["eval_k"]} - Average RPA', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(fontsize=11, loc='best')

        plt.suptitle(f'{algorithm} - Average RPA Across All Datasets',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save (one file per algorithm)
        filepath = os.path.join(CONFIG['plots_dir'], f'aggregated_rpa_{algorithm}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    Saved aggregated RPA plot: {filepath}")

    # Plot 2: Heatmap showing best strategy at 50% sampling
    # Filter for 50% sampling
    data_50 = results_df[results_df['sampling_rate'] == 50]

    # Only create heatmap if we have data at 50% sampling
    if len(data_50) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Pivot for heatmap (NDCG as metric)
        heatmap_data = data_50.pivot_table(
            index='dataset',
            columns='algorithm',
            values='ndcg',
            aggfunc='max'
        )

        # Check if heatmap_data is not empty
        if not heatmap_data.empty and heatmap_data.size > 0:
            sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd',
                        ax=ax, cbar_kws={'label': f'NDCG@{CONFIG["eval_k"]}'})
            ax.set_title(f'Best NDCG@{CONFIG["eval_k"]} at 50% Sampling (by Dataset × Algorithm)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')

            plt.tight_layout()

            filepath = os.path.join(CONFIG['plots_dir'], 'heatmap_50pct_sampling.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved heatmap plot: {filepath}")
        else:
            print(f"  Skipping heatmap (no valid data at 50% sampling)")
            plt.close()
    else:
        print(f"  Skipping heatmap (no experiments at 50% sampling rate)")


# =============================================================================
# RESULTS SAVING/LOADING
# =============================================================================

def save_results(results_df, dataset_name=None, algorithm=None, filename=None):
    """
    Save results to CSV.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results to save
    dataset_name : str, optional
        Dataset name (for specific file)
    algorithm : str, optional
        Algorithm name (for specific file)
    filename : str, optional
        Custom filename
    """
    if filename:
        filepath = os.path.join(CONFIG['results_dir'], filename)
    elif dataset_name and algorithm:
        filepath = os.path.join(CONFIG['results_dir'], f'{dataset_name}_{algorithm}.csv')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(CONFIG['results_dir'], f'results_{timestamp}.csv')

    results_df.to_csv(filepath, index=False)
    print(f"    Saved results to: {filepath}")


def load_all_results():
    """
    Load all result CSV files from results directory.

    Returns:
    --------
    pd.DataFrame : Combined results
    """
    all_files = [f for f in os.listdir(CONFIG['results_dir']) if f.endswith('.csv')]

    if not all_files:
        return pd.DataFrame()

    dfs = []
    for file in all_files:
        filepath = os.path.join(CONFIG['results_dir'], file)
        df = pd.read_csv(filepath)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_inter_file(dataset_name, data_path='dataset/'):
    """
    Load existing .inter file directly into a pandas DataFrame.
    Handles both explicit (with ratings) and implicit (no ratings) feedback datasets.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    data_path : str
        Path to dataset directory

    Returns:
    --------
    pd.DataFrame : Dataset with columns [user_id, item_id, rating, timestamp]
    """
    inter_file = os.path.join(data_path, dataset_name, f'{dataset_name}.inter')

    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"Dataset file not found: {inter_file}")

    print(f"  Loading from: {inter_file}")

    # Read the .inter file (tab-separated, first row is header with type annotations)
    df = pd.read_csv(inter_file, sep='\t')

    # RecBole .inter files have headers like "user_id:token", "item_id:token", etc.
    # Remove the type annotations from column names
    df.columns = [col.split(':')[0] for col in df.columns]

    print(f"  Loaded {len(df):,} interactions")
    print(f"  Columns: {list(df.columns)}")

    # Normalize column names: map common alternative names to standard names
    column_mappings = {
        'artist_id': 'item_id',    # lastfm
        'movie_id': 'item_id',     # movielens variants
        'book_id': 'item_id',      # book datasets
        'product_id': 'item_id',   # e-commerce datasets
        'track_id': 'item_id',     # music datasets
        'song_id': 'item_id',      # music datasets
    }

    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            print(f"  Mapping '{old_name}' → '{new_name}'")
            df = df.rename(columns={old_name: new_name})

    # Handle implicit feedback datasets (no rating column)
    if 'rating' not in df.columns:
        print(f"  No rating column found - treating as implicit feedback (all ratings = 1.0)")
        df['rating'] = 1.0

    # Handle missing timestamp (use sequential order)
    if 'timestamp' not in df.columns:
        print(f"  No timestamp column found - using sequential order")
        df['timestamp'] = range(len(df))

    # Ensure rating is numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Ensure we have all required columns
    required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after processing: {missing_cols}")

    return df


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """
    Main function to orchestrate all experiments.
    """
    print("=" * 80)
    print("Multi-Dataset RS Experiment Script with Relative Performance Analysis")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup
    print("Setting up...")
    setup_directories()
    np.random.seed(CONFIG['random_seed'])
    init_seed(CONFIG['random_seed'], True)
    print()

    # Print configuration
    print("Configuration:")
    print(f"  Datasets: {CONFIG['datasets']}")
    print(f"  Algorithms: {CONFIG['algorithms']}")
    print(f"  Sampling rates: {CONFIG['sampling_rates']}")
    print(f"  Strategies: {CONFIG['strategies']}")
    print(f"  Evaluation: {CONFIG['eval_k']}")
    print()

    # Run experiments
    all_results = []

    for dataset_name in CONFIG['datasets']:
        print(f"{'=' * 80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 80}")

        # Load dataset from existing .inter file
        print(f"Loading {dataset_name}...")
        df = load_inter_file(dataset_name)

        # Preprocess
        print(f"Preprocessing...")
        df = preprocess_data(
            df,
            min_r=CONFIG['min_ratings'],
            max_n=CONFIG['max_users'],
            min_total_ratings=CONFIG['min_total_ratings_per_user']
        )

        n_users = df['user_id'].nunique()
        n_items = df['item_id'].nunique()
        n_ratings = len(df)
        sparsity = 1 - (n_ratings / (n_users * n_items))

        print(f"  Users: {n_users:,}")
        print(f"  Items: {n_items:,}")
        print(f"  Ratings: {n_ratings:,}")
        print(f"  Sparsity: {sparsity:.4f}")
        print()

        # Compute/load difficult ratings
        difficult_data = compute_difficult_ratings(df, dataset_name)
        print()

        # Run experiments for each algorithm
        for algorithm in CONFIG['algorithms']:
            print(f"  {'~' * 76}")
            print(f"  Algorithm: {algorithm}")
            print(f"  {'~' * 76}")

            results_df = run_experiments_for_dataset(
                dataset_name, algorithm, difficult_data
            )

            # Save results
            save_results(results_df, dataset_name, algorithm)
            all_results.append(results_df)

            # Calculate RPA
            results_df = calculate_rpa(results_df)

            # Generate plots
            print(f"    Generating visualizations...")
            plot_metrics(results_df, dataset_name, algorithm)
            plot_rpa(results_df, dataset_name, algorithm)
            print()

    # Aggregate results
    print(f"{'=' * 80}")
    print("Generating aggregated analysis...")
    print(f"{'=' * 80}")

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = calculate_rpa(combined_results)

    # Save combined results
    save_results(combined_results, filename='all_results_summary.csv')

    # Generate aggregated plots
    plot_aggregated_comparison(combined_results)

    print()
    print(f"{'=' * 80}")
    print("Experiments completed successfully!")
    print(f"{'=' * 80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Results saved to:")
    print(f"  - CSV files: {CONFIG['results_dir']}")
    print(f"  - Plots: {CONFIG['plots_dir']}")
    print(f"  - Cache: {CONFIG['cache_dir']}")
    print()

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 80)
    for dataset in CONFIG['datasets']:
        for algorithm in CONFIG['algorithms']:
            data = combined_results[(combined_results['dataset'] == dataset) &
                                   (combined_results['algorithm'] == algorithm) &
                                   (combined_results['sampling_rate'] == 100)]

            if len(data) > 0:
                print(f"{dataset} - {algorithm}:")
                for strategy in CONFIG['strategies']:
                    strat_data = data[data['strategy'] == strategy]
                    if len(strat_data) > 0:
                        row = strat_data.iloc[0]
                        print(f"  {strategy:20s}: P@{CONFIG['eval_k']}={row['precision']:.4f}, "
                              f"NDCG@{CONFIG['eval_k']}={row['ndcg']:.4f}, "
                              f"MAP@{CONFIG['eval_k']}={row['map']:.4f}")
                print()


if __name__ == '__main__':
    main()
