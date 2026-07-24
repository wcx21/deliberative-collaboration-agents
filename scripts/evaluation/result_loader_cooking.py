"""
Result Loader for Experiments

Loads all experimental results into a pandas DataFrame for easy analysis.
Supports:
- Multiple models and levels
- With/without tools categories
- Oracle baseline
- Extended metrics (SEA, VEA, Hallucination Rate)
"""

import pickle as pkl
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set up paths relative to this file
from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.data_generation.cooking.load_games import load_games


# ============================================================================
# Core Result Loading Functions
# ============================================================================

def load_results_cook(result_path: str, games: Optional[List] = None, 
                      start_game_id: int = 0, end_game_id: int = 60) -> Dict[str, List]:
    """
    Load results for cook scenario.
    
    Args:
        result_path: Path to result directory containing game_* folders
        games: Optional list of game objects for discounted calculation
        start_game_id: Start game ID (inclusive, default: 0)
        end_game_id: End game ID (exclusive, default: 60)
        
    Returns:
        Dictionary with keys: 'nr', 'npr', 'npr2', 'vr', 'n_dishes', 'game_ids'
    """
    if not os.path.exists(result_path):
        return {'nr': [], 'npr': [], 'npr2': [], 'vr': [], 'n_dishes': [], 'game_ids': []}
    
    game_folders = sorted([f for f in os.listdir(result_path) if f.startswith('game_')],
                         key=lambda x: int(x.split('_')[-1]))
    
    nrs, nfrs, npr2s, vrs, n_dishes, game_ids = [], [], [], [], [], []
    
    for folder in game_folders:
        game_id = int(folder.split('_')[-1])
        
        # Filter by game_id range
        if game_id < start_game_id or game_id >= end_game_id:
            continue
        
        result_file = os.path.join(result_path, folder, 'short_result.pkl')
        if not os.path.exists(result_file):
            continue
            
        try:
            with open(result_file, 'rb') as f:
                result = pkl.load(f)
            
            final_menu = result['final_proposal']
            scores = result['scores']
            
            # Cook scenario: scores = (agent_score, flexible_score, max_reward, nr, npr)
            if len(scores) >= 5:
                nr = scores[-2]  # normalized reward
                npr = scores[-1]  # normalized adapted reward
            else:
                # Fallback for older format
                max_reward = scores[2] if len(scores) > 2 else 1.0
                nr = scores[0] / max_reward if max_reward > 0 else 0
                npr = scores[1] / max_reward if max_reward > 0 else 0
            
            # Calculate discounted reward if games provided
            npr2 = npr
            if games is not None:
                if game_id < len(games):
                    game = games[game_id]
                    try:
                        fixed_menu = game.get_feasible_sub_menu(final_menu, verbose=False)
                        discount_factor = len(fixed_menu) / len(final_menu) if len(final_menu) > 0 else 0
                        npr2 = npr * discount_factor
                    except:
                        pass
            
            vr = 1 if nr == npr and not (nr == 0 and npr == 0) else 0
            
            # Scale to (0, 100) to match task_allo scale
            nrs.append(nr * 100)
            nfrs.append(npr * 100)
            npr2s.append(npr2 * 100)
            vrs.append(vr)
            n_dishes.append(len(final_menu))
            game_ids.append(game_id)
            
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    return {
        'nr': nrs,
        'npr': nfrs,
        'npr2': npr2s,
        'vr': vrs,
        'n_dishes': n_dishes,
        'game_ids': game_ids
    }


def load_estimation_from_full_record_cook(result_path: str, game_id: int) -> Optional[Dict]:
    """
    Load estimation data from full_record.pkl for cook scenario.
    
    Args:
        result_path: Path to result directory containing game_* folders
        game_id: Game ID
        
    Returns:
        Dictionary with 'estimated_total_ingredients', 'estimated_overall_preference',
        and 'total_output_token_count', or None if not found
    """
    game_folder = f"game_{game_id}"
    full_record_path = os.path.join(result_path, game_folder, 'full_record.pkl')
    
    if not os.path.exists(full_record_path):
        return None
    
    try:
        with open(full_record_path, 'rb') as f:
            records = pkl.load(f)
        
        # records is a list, get the last record (most recent run)
        if not records or len(records) == 0:
            return None
        
        record = records[-1]
        
        # Load total_output_token_count from the first element of records list
        total_output_token_count = None
        if len(records) > 0 and isinstance(records[0], dict):
            total_output_token_count = records[0].get('total_output_token_count')
        
        # Find the last round
        round_keys = [k for k in record.keys() if k.startswith('round ')]
        if not round_keys:
            # If no rounds found, still return token count if available
            if total_output_token_count is not None:
                return {
                    'estimated_total_ingredients': None,
                    'estimated_overall_preference': None,
                    'total_output_token_count': total_output_token_count
                }
            return None
        
        # Sort rounds and get the last one
        last_round = sorted(round_keys, key=lambda x: int(x.split(' ')[1]))[-1]
        
        # Get the last agent's estimation (typically Agent 2/Bob)
        agent_keys = [k for k in record[last_round].keys() if k.startswith('Agent')]
        if not agent_keys:
            # If no agents found, still return token count if available
            if total_output_token_count is not None:
                return {
                    'estimated_total_ingredients': None,
                    'estimated_overall_preference': None,
                    'total_output_token_count': total_output_token_count
                }
            return None
        
        # Try Agent 2 first, then Agent 1
        for agent_name in ['Agent 2', 'Agent 1']:
            if agent_name in record[last_round]:
                observer_data = record[last_round][agent_name].get('observer_agent', {})
                if observer_data:
                    total_ingredients = observer_data.get('total_ingredients')
                    overall_preference = observer_data.get('overall_preference')
                    if total_ingredients is not None and overall_preference is not None:
                        return {
                            'estimated_total_ingredients': total_ingredients,
                            'estimated_overall_preference': overall_preference,
                            'total_output_token_count': total_output_token_count
                        }
        
        # If no estimation data found, still return token count if available
        if total_output_token_count is not None:
            return {
                'estimated_total_ingredients': None,
                'estimated_overall_preference': None,
                'total_output_token_count': total_output_token_count
            }
        
        return None
        
    except Exception as e:
        print(f"Error loading full_record for game {game_id}: {e}")
        return None


def compute_extended_metrics_for_game(result_path: str, game_id: int, game) -> Dict[str, float]:
    """
    Compute extended metrics (SEA, VEA, Hallucination Rate) for a single game.
    
    Args:
        result_path: Path to result directory
        game_id: Game ID
        game: Game object
        
    Returns:
        Dictionary with 'sea', 'vea', 'hallucination_rate', or None values if unavailable
    """
    from delib_collab.data_generation.cooking import planner as planning_algos
    
    estimation_data = load_estimation_from_full_record_cook(result_path, game_id)
    
    if not estimation_data:
        return {
            'sea': None,
            'vea': None,
            'hallucination_rate': None,
            'total_output_token_count': None
        }
    
    try:
        # Get ground truth
        best_menu = game.best_menu
        best_menu_ingredients = np.sum(game.recipes[best_menu], axis=0)
        possible_dishes = game.possible_dishes
        possible_menu_ingredients = np.sum(game.recipes[possible_dishes], axis=0)
        best_menu_ingredients = possible_menu_ingredients
        
        gt_values = (game.agent_0_values + game.agent_1_values) / 2.0
        
        # Process estimated ingredients
        estimated_ingredients = estimation_data['estimated_total_ingredients']
        # Handle both list and single dict formats
        if isinstance(estimated_ingredients, list) and len(estimated_ingredients) > 0:
            final_estimated_ingredients = estimated_ingredients[0]
        else:
            final_estimated_ingredients = estimated_ingredients
        
        final_estimated_ingredients_vec = planning_algos.vectorize_state(
            final_estimated_ingredients, game.ingredients
        )
        
        # Calculate State Estimation Accuracy
        # Calculate SEA: 1 - mean(|estimated - gt| / gt) for all ingredients
        # Direct comparison of estimated state vs ground truth state
        gt_state = game.vec_ingredients_state
        est_state = final_estimated_ingredients_vec
        
        # Calculate absolute difference
        abs_diff = np.abs(est_state - gt_state)
        
        # Normalize by ground truth (only where gt > 0 to avoid division by zero)
        # For positions where gt == 0, we only consider if estimated > 0 (hallucination)
        # For positions where gt > 0, we calculate |est - gt| / gt
        mask_gt_positive = gt_state > 0
        
        if np.any(mask_gt_positive):
            # For positions where gt > 0: calculate normalized error
            normalized_errors = abs_diff[mask_gt_positive] / gt_state[mask_gt_positive]
            # SEA = 1 - mean(normalized_error)
            state_acc = 1 - normalized_errors.mean()
        else:
            # No positive ground truth values, check if there are any hallucinations
            if np.any(est_state > 0):
                state_acc = 0.0  # All are hallucinations
            else:
                state_acc = 1.0  # Perfect match (both are zero)
        
        # Ensure SEA is in [0, 1] range
        state_acc = np.clip(state_acc, 0.0, 1.0)
        
        # Calculate Hallucination Rate: count positions where estimated > gt_vec_state
        # hall_rt = number of hallucination positions / total number of positions in gt_vec_state
        hallucination_mask = est_state > gt_state
        num_hallucinations = np.sum(hallucination_mask)
        total_positions = len(gt_state)
        
        if total_positions > 0:
            hall_rt = num_hallucinations / total_positions
        else:
            hall_rt = 0.0
        
        # Process estimated preference
        estimated_preference = estimation_data['estimated_overall_preference']
        if isinstance(estimated_preference, list) and len(estimated_preference) > 0:
            final_estimated_preference = estimated_preference[-1]
        else:
            final_estimated_preference = estimated_preference
        
        if isinstance(final_estimated_preference, str):
            import json
            final_estimated_preference = json.loads(final_estimated_preference.split('\n')[0])
        
        estimated_pref_vec = np.zeros(len(game.dishes))
        for dish_idx, dish in enumerate(game.dishes):
            estimated_pref_vec[dish_idx] = final_estimated_preference.get(dish, 0)
        
        # Calculate Value Estimation Accuracy
        gt_possible_dishes_value = gt_values[possible_dishes]
        est_possible_dishes_value = estimated_pref_vec[possible_dishes]
        
        if np.linalg.norm(est_possible_dishes_value) == 0:
            value_acc = 0
        else:
            dp = np.dot(est_possible_dishes_value, gt_possible_dishes_value)
            value_acc = dp / np.linalg.norm(est_possible_dishes_value) / np.linalg.norm(gt_possible_dishes_value)
        
        return {
            'sea': state_acc,
            'vea': value_acc,
            'hallucination_rate': hall_rt,
            'total_output_token_count': estimation_data.get('total_output_token_count')
        }
        
    except Exception as e:
        print(f"Error computing extended metrics for game {game_id}: {e}")
        return {
            'sea': None,
            'vea': None,
            'hallucination_rate': None,
            'total_output_token_count': estimation_data.get('total_output_token_count') if estimation_data else None
        }


def load_results_oracle(oracle_path: str, format: str = 'pkl') -> Dict[str, List]:
    """
    Load oracle baseline results.
    
    Args:
        oracle_path: Path to oracle.pkl file or oracle result directory
        format: 'pkl' for single file, 'dir' for directory structure
        
    Returns:
        Dictionary with same format as load_results_cook
    """
    if format == 'pkl':
        # Single oracle.pkl file format
        if not os.path.exists(oracle_path):
            return {'nr': [], 'npr': [], 'npr2': [], 'vr': [], 'n_dishes': [], 'game_ids': []}
        
        try:
            with open(oracle_path, 'rb') as f:
                oracle_result = pkl.load(f)
            
            nrs, nfrs, npr2s, vrs, n_dishes, game_ids = [], [], [], [], [], []
            
            for key, result in oracle_result.items():
                if not key.startswith('game_'):
                    continue
                
                game_id = int(key.split('_')[-1])
                final_menu = result['final_proposal']
                scores = result['scores']
                
                if len(scores) >= 5:
                    nr = scores[-2]
                    npr = scores[-1]
                else:
                    max_reward = scores[2] if len(scores) > 2 else 1.0
                    nr = scores[0] / max_reward if max_reward > 0 else 0
                    npr = scores[1] / max_reward if max_reward > 0 else 0
                
                npr2 = npr  # Oracle doesn't need discount
                vr = 1 if nr == npr and not (nr == 0 and npr == 0) else 0
                
                # Scale to (0, 100) to match task_allo scale
                nrs.append(nr * 100)
                nfrs.append(npr * 100)
                npr2s.append(npr2 * 100)
                vrs.append(vr)
                n_dishes.append(len(final_menu))
                game_ids.append(game_id)
            
            # Sort by game_id
            sorted_indices = np.argsort(game_ids)
            return {
                'nr': [nrs[i] for i in sorted_indices],
                'npr': [nfrs[i] for i in sorted_indices],
                'npr2': [npr2s[i] for i in sorted_indices],
                'vr': [vrs[i] for i in sorted_indices],
                'n_dishes': [n_dishes[i] for i in sorted_indices],
                'game_ids': [game_ids[i] for i in sorted_indices]
            }
            
        except Exception as e:
            print(f"Error loading oracle.pkl: {e}")
            return {'nr': [], 'npr': [], 'npr2': [], 'vr': [], 'n_dishes': [], 'game_ids': []}
    
    else:
        # Directory structure format (same as regular results)
        return load_results_cook(oracle_path)


# ============================================================================
# Main DataFrame Loading Function
# ============================================================================

def load_all_results_to_dataframe(
    exp_name: str,
    models: List[str],
    levels: List[int],
    start_game_id: int = 0,
    end_game_id: int = 60,
    base_dir: str = 'result',
    game_folder: str = 'test_games',
    n_games: int = 60,
    categories: Optional[List[str]] = None,
    oracle_exp_name: Optional[str] = None,
    include_extended_metrics: bool = True
) -> pd.DataFrame:
    """
    Load all experimental results into a pandas DataFrame.
    
    Args:
        exp_name: Experiment name (e.g., 'experiment_name')
        models: List of model names
        levels: List of levels to process (e.g., [1, 2])
        start_game_id: Start game ID (inclusive, default: 0)
        end_game_id: End game ID (exclusive, default: 60)
        base_dir: Base directory for results (default: 'result')
        game_folder: Game data folder name
        n_games: Number of games
        categories: List of categories (e.g., ['with_tools', 'no_tools']). If None, uses both.
        oracle_exp_name: Optional oracle experiment name. If provided, loads oracle results.
        include_extended_metrics: Whether to compute SEA, VEA, and hallucination_rate
        
    Returns:
        pandas DataFrame with columns:
        - model: Model name
        - level: Level (1 or 2)
        - category: 'with_tools' or 'no_tools'
        - is_oracle: Whether this is oracle baseline
        - game_id: Game ID
        - nr: Normalized Reward
        - nar: Normalized Adapted Reward (NPR)
        - npr2: Normalized Discounted Reward
        - vr: VR (Reward Equal Ratio)
        - n_dishes: Number of dishes
        - sea: State Estimation Accuracy (if include_extended_metrics=True)
        - vea: Value Estimation Accuracy (if include_extended_metrics=True)
        - hallucination_rate: Hallucination Rate (if include_extended_metrics=True)
        - total_output_token_count: Total output token count (if include_extended_metrics=True)
    """
    if categories is None:
        categories = ['with_tools', 'no_tools']
    
    all_rows = []
    
    # Load games for extended metrics
    games_by_level = {}
    if include_extended_metrics:
        for level in levels:
            try:
                all_games = load_games(data_root=game_folder, level=level, n_games=n_games)
                games_by_level[level] = all_games[start_game_id:end_game_id] if all_games else None
            except Exception as e:
                print(f"Warning: Failed to load games for level {level}: {e}")
                games_by_level[level] = None
    
    # Load regular model results
    for level in levels:
        games = games_by_level.get(level) if include_extended_metrics else None
        
        for category in categories:
            for model in models:
                folder_name = f"{exp_name}_{model}"
                category_path = f"level_{level}_{category}"
                result_path = os.path.join(base_dir, folder_name, category_path)
                
                if not os.path.exists(result_path):
                    print(f"Warning: Path not found: {result_path}")
                    continue
                
                # Load basic results
                results = load_results_cook(result_path, games, start_game_id, end_game_id)
                
                # Add rows to DataFrame
                for idx, game_id in enumerate(results['game_ids']):
                    row = {
                        'model': model,
                        'level': level,
                        'category': category,
                        'is_oracle': False,
                        'game_id': game_id,
                        'nr': results['nr'][idx],
                        'nar': results['npr'][idx],  # NAR = NPR
                        'npr2': results['npr2'][idx],
                        'vr': results['vr'][idx],
                        'n_dishes': results['n_dishes'][idx],
                    }
                    
                    # Add extended metrics if requested
                    if include_extended_metrics:
                        if games is not None:
                            game_idx = game_id - start_game_id
                            if 0 <= game_idx < len(games) and games[game_idx] is not None:
                                ext_metrics = compute_extended_metrics_for_game(
                                    result_path, game_id, games[game_idx]
                                )
                                row['sea'] = ext_metrics['sea']
                                row['vea'] = ext_metrics['vea']
                                row['hallucination_rate'] = ext_metrics['hallucination_rate']
                                row['total_output_token_count'] = ext_metrics['total_output_token_count']
                            else:
                                # Try to load token count even if game object is not available
                                estimation_data = load_estimation_from_full_record_cook(result_path, game_id)
                                row['sea'] = None
                                row['vea'] = None
                                row['hallucination_rate'] = None
                                row['total_output_token_count'] = estimation_data.get('total_output_token_count') if estimation_data else None
                        else:
                            # Try to load token count even if games are not loaded
                            estimation_data = load_estimation_from_full_record_cook(result_path, game_id)
                            row['sea'] = None
                            row['vea'] = None
                            row['hallucination_rate'] = None
                            row['total_output_token_count'] = estimation_data.get('total_output_token_count') if estimation_data else None
                    else:
                        row['sea'] = None
                        row['vea'] = None
                        row['hallucination_rate'] = None
                        row['total_output_token_count'] = None
                    
                    all_rows.append(row)
    
    # Load oracle results if specified
    # For each model, load its corresponding oracle version
    if oracle_exp_name:
        for level in levels:
            for category in categories:
                for model in models:
                    folder_name = f"{oracle_exp_name}_{model}"
                    category_path = f"level_{level}_{category}"
                    oracle_path = os.path.join(base_dir, folder_name, category_path)
                    
                    # Try directory format first
                    if not os.path.exists(oracle_path):
                        # Try pkl format
                        oracle_pkl_path = os.path.join(base_dir, folder_name, 'oracle.pkl')
                        if os.path.exists(oracle_pkl_path):
                            oracle_results = load_results_oracle(oracle_pkl_path, format='pkl')
                        else:
                            print(f"Warning: Oracle path not found for {model}: {oracle_path}")
                            continue
                    else:
                        oracle_results = load_results_oracle(oracle_path, format='dir')
                    
                    # Add oracle rows (same model name, but is_oracle=True)
                    for idx, game_id in enumerate(oracle_results['game_ids']):
                        # Filter by game_id range
                        if game_id < start_game_id or game_id >= end_game_id:
                            continue
                        
                        row = {
                            'model': model,  # Same model name as regular version
                            'level': level,
                            'category': category,
                            'is_oracle': True,  # Mark as oracle version
                            'game_id': game_id,
                            'nr': oracle_results['nr'][idx],
                            'nar': oracle_results['npr'][idx],
                            'npr2': oracle_results['npr2'][idx],
                            'vr': oracle_results['vr'][idx],
                            'n_dishes': oracle_results['n_dishes'][idx],
                        }
                        
                        # Oracle doesn't have SEA/VEA, set to None
                        row['sea'] = None
                        row['vea'] = None
                        row['hallucination_rate'] = None
                        row['total_output_token_count'] = None
                        
                        all_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_rows)
    
    # Sort by level, category, model, game_id
    if len(df) > 0:
        df = df.sort_values(['level', 'category', 'model', 'game_id']).reset_index(drop=True)
    
    return df


# ============================================================================
# Simple Statistics Function
# ============================================================================

def print_simple_statistics(df: pd.DataFrame, group_by: Optional[List[str]] = None) -> None:
    """
    Print simple statistics grouped by specified columns.
    
    Args:
        df: DataFrame from load_all_results_to_dataframe
        group_by: List of columns to group by (e.g., ['model', 'level', 'category'])
                 If None, groups by ['model', 'level', 'category']
    """
    if group_by is None:
        group_by = ['model', 'level', 'category']
    
    # Filter out oracle if needed (or include it)
    metrics = ['nr', 'nar', 'vr']
    if 'sea' in df.columns:
        metrics.append('sea')
    if 'vea' in df.columns:
        metrics.append('vea')
    
    print("\n" + "="*80)
    print("Simple Statistics")
    print("="*80)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        print(f"\n{metric.upper()}:")
        print("-"*80)
        
        grouped = df.groupby(group_by)[metric].agg(['mean', 'std', 'count'])
        grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
        
        for idx, row in grouped.iterrows():
            if isinstance(idx, tuple):
                group_str = " | ".join([f"{col}={val}" for col, val in zip(group_by, idx)])
            else:
                group_str = f"{group_by[0]}={idx}"
            
            print(f"  {group_str}: {row['mean']:.4f} ± {row['se']:.4f} (n={int(row['count'])})")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Load experimental results into DataFrame')
    parser.add_argument('-s', '--start', type=int, default=0, 
                       help='Start game ID (inclusive, default: 0)')
    parser.add_argument('-e', '--end', type=int, default=60,
                       help='End game ID (exclusive, default: 60)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Example usage
    root_exp_name = 'experiment_name'
    oracle_exp_name = 'experiment_name_oracle'
    
    models = [
        'gpt-5.1',
        'deepseek-v3.2',
        'glm-4.7',
        'gpt-4.1-mini',
        'qwen3-next-80b',
        'qwen3-32b',
        'qwen3-30b',
    ]
    
    # Load all results
    print("Loading all results...")
    df = load_all_results_to_dataframe(
        exp_name=root_exp_name,
        models=models,
        levels=[1, 2],
        start_game_id=args.start,
        end_game_id=args.end,
        base_dir='result',
        game_folder='test_games',
        n_games=60,
        categories=['with_tools', 'no_tools'],
        oracle_exp_name=oracle_exp_name,
        include_extended_metrics=True
    )
    
    print(f"\nLoaded {len(df)} rows")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    # Print simple statistics
    print_simple_statistics(df)
    
    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDataFrame saved to: {args.output}")
    else:
        # Default output file
        output_file = os.path.join('result', f'results_{args.start}_{args.end}.csv')
        df.to_csv(output_file, index=False)
        print(f"\nDataFrame saved to: {output_file}")