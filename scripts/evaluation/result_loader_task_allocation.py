"""
Result Loader for Task Allocation Experiments

Loads all experimental results into a pandas DataFrame for easy analysis.
This module is dedicated to task_allo scenario only.
For cook scenario, use stat_result_loader.py.

Supports:
- Multiple models and levels (typically level 1)
- With/without tools categories
- Extended metrics (SEA, VEA, Hallucination Rate)
- Oracle baseline (exp_name format: {exp_name}_oracle, path format: baseline_{category})
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

# Note: This module is dedicated to task_allo scenario only
# For cook scenario, use stat_result_loader.py


# ============================================================================
# Core Result Loading Functions
# ============================================================================

def load_estimation_from_full_record_task_allo(result_path: str, game_id: int) -> Optional[Dict]:
    """
    Load estimation data from full_record.pkl for task allocation scenario.
    
    Args:
        result_path: Path to result directory containing game_* folders
        game_id: Game ID
        
    Returns:
        Dictionary with 'estimated_total_resources', 'estimated_overall_preferences',
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
                    'estimated_total_resources': None,
                    'estimated_overall_preferences': None,
                    'total_output_token_count': total_output_token_count
                }
            return None
        
        # Sort rounds and get the last one
        last_round = sorted(round_keys, key=lambda x: int(x.split(' ')[1]))[-1]
        
        # Get the last agent's estimation (typically agent_2/Leader)
        agent_keys = [k for k in record[last_round].keys() if k.startswith('agent_')]
        if not agent_keys:
            # If no agents found, still return token count if available
            if total_output_token_count is not None:
                return {
                    'estimated_total_resources': None,
                    'estimated_overall_preferences': None,
                    'total_output_token_count': total_output_token_count
                }
            return None
        
        # Try agent_2 first, then agent_1, then agent_0
        for agent_name in ['agent_2', 'agent_1', 'agent_0']:
            if agent_name in record[last_round]:
                observer_data = record[last_round][agent_name].get('observer_agent', {})
                if observer_data:
                    total_resources = observer_data.get('total_resources')
                    overall_preferences = observer_data.get('overall_preferences')
                    if total_resources is not None and overall_preferences is not None:
                        return {
                            'estimated_total_resources': total_resources,
                            'estimated_overall_preferences': overall_preferences,
                            'total_output_token_count': total_output_token_count
                        }
        
        # If no estimation data found, still return token count if available
        if total_output_token_count is not None:
            return {
                'estimated_total_resources': None,
                'estimated_overall_preferences': None,
                'total_output_token_count': total_output_token_count
            }
        
        return None
        
    except Exception as e:
        print(f"Error loading full_record for game {game_id}: {e}")
        return None


def compute_extended_metrics_for_game_task_allo(result_path: str, game_id: int, game) -> Dict[str, float]:
    """
    Compute extended metrics (SEA, VEA, Hallucination Rate) for a single task allocation game.
    
    SEA Definition (aligned with cook task):
    State Estimation Accuracy (SEA): The percentage of influential resources correctly estimated 
    when making the final proposal. A resource is considered "influential" if it is used in the 
    optimal allocation.
    
    Args:
        result_path: Path to result directory
        game_id: Game ID
        game: TaskAllocationGame object
        
    Returns:
        Dictionary with 'sea', 'vea', 'hallucination_rate', or None values if unavailable
    """
    try:
        from delib_collab.data_generation.task_allocation.planner import vectorize_task_allocation_state
    except ImportError:
        print("Warning: Could not import task_allocation_planning_algos")
        return {
            'sea': None,
            'vea': None,
            'hallucination_rate': None
        }
    
    estimation_data = load_estimation_from_full_record_task_allo(result_path, game_id)
    
    if not estimation_data:
        return {
            'sea': None,
            'vea': None,
            'hallucination_rate': None,
            'total_output_token_count': None
        }
    
    try:
        # Get ground truth resource state
        gt_resource_state = game.game_resource_state
        gt_vec_state = game.vec_resource_state
        
        # Get best allocation and its resource requirements
        best_allocation = game.best_allocation
        constraint_matrix = game.constraint_matrix
        
        # Calculate resource requirements for best allocation (influential resources)
        if best_allocation is not None and np.sum(best_allocation) > 0:
            # best_allocation_resource_usage: resources used by the optimal allocation
            best_allocation_resource_usage = constraint_matrix @ best_allocation
        else:
            # No valid allocation, return 0
            return {
                'sea': 0.0,
                'vea': None,
                'hallucination_rate': None
            }
        
        # Identify influential resources (resources used in optimal allocation)
        influential_mask = best_allocation_resource_usage > 0
        
        if not np.any(influential_mask):
            # No influential resources
            return {
                'sea': 0.0,
                'vea': None,
                'hallucination_rate': None
            }
        
        # Process estimated resources
        estimated_resources = estimation_data['estimated_total_resources']
        # Handle both dict and nested formats
        if isinstance(estimated_resources, dict):
            agent_private_resources = estimated_resources.get('agent_private_resources', {})
            public_resources = estimated_resources.get('public_resources', {})
        else:
            return {
                'sea': None,
                'vea': None,
                'hallucination_rate': None
            }
        
        # Vectorize estimated resources
        estimated_vec_state = vectorize_task_allocation_state(
            agent_private_resources,
            public_resources,
            game.private_resource_list,
            game.public_resource_list
        )
        
        # Calculate State Estimation Accuracy (SEA)
        # Direct comparison of estimated state vs ground truth state
        gt_state = gt_vec_state
        est_state = estimated_vec_state
        
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
            sea = 1 - normalized_errors.mean()
        else:
            # No positive ground truth values, check if there are any hallucinations
            if np.any(est_state > 0):
                sea = 0.0  # All are hallucinations
            else:
                sea = 1.0  # Perfect match (both are zero)
        
        # Ensure SEA is in [0, 1] range
        sea = np.clip(sea, 0.0, 1.0)
        
        # Calculate Hallucination Rate: count positions where estimated > gt_vec_state
        # hall_rt = number of hallucination positions / total number of positions in gt_vec_state
        hallucination_mask = estimated_vec_state > gt_vec_state
        num_hallucinations = np.sum(hallucination_mask)
        total_positions = len(gt_vec_state)
        
        if total_positions > 0:
            hall_rt = num_hallucinations / total_positions
        else:
            hall_rt = 0.0
        
        # Process estimated preferences for VEA
        estimated_preferences = estimation_data['estimated_overall_preferences']
        # Handle both dict and nested formats
        if isinstance(estimated_preferences, dict):
            # Convert to value matrix format
            num_agents = len(game.agents)
            num_tasks = len(game.tasks)
            estimated_value_matrix = np.zeros((num_agents, num_tasks))
            
            for agent_idx, agent_name in enumerate(game.agents):
                agent_prefs = estimated_preferences.get(agent_name, {})
                for task_idx, task_name in enumerate(game.tasks):
                    estimated_value_matrix[agent_idx, task_idx] = agent_prefs.get(task_name, 0)
        else:
            estimated_value_matrix = np.zeros((len(game.agents), len(game.tasks)))
        
        # Get ground truth value matrix
        gt_value_matrix = game.value_matrix
        
        # Calculate Value Estimation Accuracy (cosine similarity on value matrices)
        gt_values_flat = gt_value_matrix.flatten()
        est_values_flat = estimated_value_matrix.flatten()
        
        if np.linalg.norm(est_values_flat) == 0:
            value_acc = 0
        else:
            dp = np.dot(est_values_flat, gt_values_flat)
            value_acc = dp / np.linalg.norm(est_values_flat) / np.linalg.norm(gt_values_flat)
        
        return {
            'sea': float(sea),
            'vea': float(value_acc),
            'hallucination_rate': float(hall_rt),
            'total_output_token_count': estimation_data.get('total_output_token_count')
        }
        
    except Exception as e:
        print(f"Error computing extended metrics for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'sea': None,
            'vea': None,
            'hallucination_rate': None,
            'total_output_token_count': estimation_data.get('total_output_token_count') if estimation_data else None
        }


def load_results_task_allo(result_path: str, games: Optional[List] = None,
                            start_game_id: int = 0, end_game_id: int = 60) -> Dict[str, List]:
    """
    Load results for task allocation scenario.
    
    Args:
        result_path: Path to result directory containing game_* folders
        games: Optional list of game objects (not used currently, reserved for future use)
        start_game_id: Start game ID (inclusive, default: 0)
        end_game_id: End game ID (exclusive, default: 60)
        
    Returns:
        Dictionary with keys: 'nr', 'nar', 'vr', 'n_tasks', 'game_ids'
    """
    if not os.path.exists(result_path):
        return {'nr': [], 'nar': [], 'vr': [], 'n_tasks': [], 'game_ids': []}
    
    game_folders = sorted([f for f in os.listdir(result_path) if f.startswith('game_')],
                         key=lambda x: int(x.split('_')[-1]))
    
    nrs, nars, vrs, n_tasks, game_ids = [], [], [], [], []
    
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
            
            final_allocation = result['final_proposal']
            
            # Priority: read from evaluation_metrics (more reliable)
            if 'evaluation_metrics' in result:
                metrics = result['evaluation_metrics']
                nr = metrics.get('nr', 0.0)  # Normalized Reward (percentage, 0-100)
                nar = metrics.get('nar', 0.0)  # Normalized Adjusted Reward (percentage, 0-100)
            else:
                # Compatible with new format: from scores
                scores = result.get('scores', ())
                if len(scores) >= 5:
                    # New format: (agent_score, max_reward, score_ratio, nr, nar)
                    nr = scores[-2]  # NR (percentage, 0-100)
                    nar = scores[-1]  # NAR (percentage, 0-100)
                elif len(scores) >= 3:
                    # Old format: (agent_score, max_reward, score_ratio)
                    score_ratio = scores[-1]
                    nr = score_ratio * 100  # Convert to percentage
                    nar = score_ratio * 100
                else:
                    # Fallback: calculate from scores
                    if len(scores) >= 2 and scores[1] > 0:
                        score_ratio = scores[0] / scores[1]
                        nr = score_ratio * 100
                        nar = score_ratio * 100
                    else:
                        nr = 0.0
                        nar = 0.0
            
            # VR (Valid Ratio): NR == NAR and not both zero
            vr = 1 if nr == nar and not (nr == 0 and nar == 0) else 0
            
            nrs.append(nr)
            nars.append(nar)
            vrs.append(vr)
            n_tasks.append(len(final_allocation))
            game_ids.append(game_id)
            
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue
    
    return {
        'nr': nrs,
        'nar': nars,  # NAR (Normalized Adjusted Reward)
        'vr': vrs,
        'n_tasks': n_tasks,
        'game_ids': game_ids
    }


# ============================================================================
# Main DataFrame Loading Function
# ============================================================================

def load_all_results_to_dataframe(
    exp_name: str = 'experiment_name',
    models: List[str] = None,
    levels: List[int] = None,
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
    Load all experimental results into a pandas DataFrame for task_allo scenario.
    
    This function is dedicated to task_allo scenario only. For cook scenario, use stat_result_loader.py.
    
    Args:
        exp_name: Experiment name (default: 'experiment_name')
        models: List of model names (default: None, will use default models if not provided)
        levels: List of levels to process (default: None, will use [1] if not provided)
        start_game_id: Start game ID (inclusive, default: 0)
        end_game_id: End game ID (exclusive, default: 60)
        base_dir: Base directory for results (default: 'result')
        game_folder: Game data folder name (default: 'test_games')
        n_games: Number of games
        categories: List of categories (e.g., ['with_tools', 'no_tools']). If None, uses both.
        oracle_exp_name: Optional oracle experiment name. If provided, loads oracle results.
                         Oracle exp_name should be {exp_name}_oracle, and paths use baseline_{category}
        include_extended_metrics: Whether to compute SEA, VEA, and hallucination_rate
        
    Returns:
        pandas DataFrame with columns:
        - model: Model name
        - level: Level (typically 1 for task_allo)
        - category: 'with_tools' or 'no_tools'
        - is_oracle: Whether this is oracle baseline
        - game_id: Game ID
        - nr: Normalized Reward
        - nar: Normalized Adapted Reward
        - vr: VR (Reward Equal Ratio)
        - n_tasks: Number of tasks
        - sea: State Estimation Accuracy (if include_extended_metrics=True)
        - vea: Value Estimation Accuracy (if include_extended_metrics=True)
        - hallucination_rate: Hallucination Rate (if include_extended_metrics=True)
        - total_output_token_count: Total output token count (if include_extended_metrics=True)
    """
    if categories is None:
        categories = ['with_tools', 'no_tools']
    if levels is None:
        levels = [1]
    if models is None:
        models = [
            'gpt-5.1',
            'deepseek-v3.2',
            'glm-4.7',
            'gpt-4.1-mini',
            'qwen3-next-80b',
            'qwen3-32b',
            'qwen3-30b',
        ]
    
    all_rows = []
    
    # Task allocation specific loading functions
    load_results_fn = load_results_task_allo
    compute_extended_metrics_fn = compute_extended_metrics_for_game_task_allo
    
    # Load task-allocation games from the cleaned data layout.
    load_games_fn = None
    try:
        from delib_collab.data_generation.task_allocation.io import load_games as load_games_task_allo
        load_games_fn = lambda: load_games_task_allo(target_path=game_folder, game_batch_name='level_1_and_2', n_games=n_games)
    except ImportError as e:
        print(f"Warning: Could not import task allocation game loader: {e}")
        load_games_fn = None
    
    # Load games for extended metrics
    games_by_level = {}
    if include_extended_metrics:
        try:
            if load_games_fn:
                all_games = load_games_fn()
                # For task allocation, use the same games for all levels
                for level in levels:
                    games_by_level[level] = all_games[start_game_id:end_game_id] if all_games else None
        except Exception as e:
            print(f"Warning: Failed to load games: {e}")
            for level in levels:
                games_by_level[level] = None
    
    # Load regular model results
    for level in levels:
        games = games_by_level.get(level) if include_extended_metrics else None
        
        for category in categories:
            for model in models:
                folder_name = f"{exp_name}_{model}"
                # Build category path for task_allo
                category_path = f"task_allocation_level_{level}_{category}"
                result_path = os.path.join(base_dir, folder_name, category_path)
                
                if not os.path.exists(result_path):
                    print(f"Warning: Path not found: {result_path}")
                    continue
                
                # Load basic results
                results = load_results_fn(result_path, games, start_game_id, end_game_id)
                
                # Add rows to DataFrame
                for idx, game_id in enumerate(results['game_ids']):
                    row = {
                        'model': model,
                        'level': level,
                        'category': category,
                        'is_oracle': False,
                        'game_id': game_id,
                        'nr': results['nr'][idx],
                        'nar': results.get('nar', [])[idx],
                        'vr': results['vr'][idx],
                    }
                    
                    # Add task_allo-specific columns
                    row['n_tasks'] = results.get('n_tasks', [])[idx] if idx < len(results.get('n_tasks', [])) else 0
                    
                    # Add extended metrics if requested
                    if include_extended_metrics:
                        game_obj = None
                        
                        # Try to get game object from batch-loaded games
                        if games is not None:
                            game_idx = game_id - start_game_id
                            if 0 <= game_idx < len(games) and games[game_idx] is not None:
                                game_obj = games[game_idx]
                        
                        # If batch loading failed, try to load individual game
                        if game_obj is None:
                            try:
                                from delib_collab.data_generation.task_allocation.io import load_games as load_games_task_allo
                                
                                # Try to load just this one game
                                # Note: game files are named game_0000.pkl, game_0001.pkl, etc.
                                single_game_list = load_games_task_allo(
                                    target_path=game_folder,
                                    game_batch_name='level_1_and_2',
                                    n_games=None,
                                    game_idxes=[game_id]
                                )
                                if single_game_list and len(single_game_list) > 0:
                                    game_obj = single_game_list[0]
                                else:
                                    # Fallback: try with relative path
                                    single_game_list = load_games_task_allo(
                                        target_path=game_folder,
                                        game_batch_name='level_1_and_2',
                                        n_games=None,
                                        game_idxes=[game_id]
                                    )
                                    if single_game_list and len(single_game_list) > 0:
                                        game_obj = single_game_list[0]
                            except Exception as e:
                                # Debug: print error for first few games only to avoid spam
                                if game_id < 3:
                                    print(f"  Warning: Failed to load individual game {game_id}: {e}")
                                pass
                        
                        # Compute extended metrics if we have game object
                        if game_obj is not None:
                            ext_metrics = compute_extended_metrics_fn(
                                result_path, game_id, game_obj
                            )
                            row['sea'] = ext_metrics['sea']
                            row['vea'] = ext_metrics['vea']
                            row['hallucination_rate'] = ext_metrics['hallucination_rate']
                            row['total_output_token_count'] = ext_metrics['total_output_token_count']
                        else:
                            # Try to load token count even if game object is not available
                            estimation_data = load_estimation_from_full_record_task_allo(result_path, game_id)
                            row['sea'] = None
                            row['vea'] = None
                            row['hallucination_rate'] = None
                            row['total_output_token_count'] = estimation_data.get('total_output_token_count') if estimation_data else None
                    else:
                        # Try to load token count even if extended metrics are not requested
                        estimation_data = load_estimation_from_full_record_task_allo(result_path, game_id)
                        row['sea'] = None
                        row['vea'] = None
                        row['hallucination_rate'] = None
                        row['total_output_token_count'] = estimation_data.get('total_output_token_count') if estimation_data else None
                    
                    all_rows.append(row)
    
    # Load oracle results if specified
    # For each model, load its corresponding oracle version
    if oracle_exp_name:
        for level in levels:
            for category in categories:
                for model in models:
                    folder_name = f"{oracle_exp_name}_{model}"
                    # Oracle path format: task_allocation_level_{level}_baseline_{category}
                    category_path = f"task_allocation_level_{level}_baseline_{category}"
                    oracle_path = os.path.join(base_dir, folder_name, category_path)
                    
                    if not os.path.exists(oracle_path):
                        print(f"Warning: Oracle path not found for {model}: {oracle_path}")
                        continue
                    
                    # Load oracle results using the same function as regular results
                    oracle_results = load_results_fn(oracle_path, None, start_game_id, end_game_id)
                    
                    # Add oracle rows (same model name, but is_oracle=True)
                    for idx, game_id in enumerate(oracle_results['game_ids']):
                        # Filter by game_id range
                        if game_id < start_game_id or game_id >= end_game_id:
                            continue
                        
                        row = {
                            'model': model,  # Same model name as regular version
                            'level': level,
                            'category': category,  # Keep original category name (without baseline_)
                            'is_oracle': True,  # Mark as oracle version
                            'game_id': game_id,
                            'nr': oracle_results['nr'][idx],
                            'nar': oracle_results.get('nar', [])[idx],
                            'vr': oracle_results['vr'][idx],
                        }
                        
                        # Add task_allo-specific columns
                        row['n_tasks'] = oracle_results.get('n_tasks', [])[idx] if idx < len(oracle_results.get('n_tasks', [])) else 0
                        
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
        
        # Print statistics about extended metrics availability
        if include_extended_metrics and 'sea' in df.columns:
            total_rows = len(df)
            sea_available = df['sea'].notna().sum()
            vea_available = df['vea'].notna().sum()
            hall_available = df['hallucination_rate'].notna().sum()
            
            if total_rows > 0:
                print(f"\nExtended Metrics Availability:")
                print(f"  Total games: {total_rows}")
                print(f"  SEA available: {sea_available} ({sea_available/total_rows*100:.1f}%)")
                print(f"  VEA available: {vea_available} ({vea_available/total_rows*100:.1f}%)")
                print(f"  Hallucination Rate available: {hall_available} ({hall_available/total_rows*100:.1f}%)")
                
                if sea_available < total_rows:
                    print(f"\n  Note: {total_rows - sea_available} games missing full_record.pkl or game objects")
                    print(f"        These games will have None for extended metrics but basic metrics are still recorded.")
    
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
    parser.add_argument('result_path', type=str, nargs='?', default=None,
                       help='Path to result directory (optional, for auto-detection)')
    parser.add_argument('-s', '--start', type=int, default=0, 
                       help='Start game ID (inclusive, default: 0)')
    parser.add_argument('-e', '--end', type=int, default=60,
                       help='End game ID (exclusive, default: 60)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output CSV file path (optional)')
    parser.add_argument('--scenario', type=str, default='task_allo', choices=['task_allo'],
                       help='Scenario type: task_allo (this loader is dedicated to task_allo only)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (optional, extracted from path if not specified)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (optional, for single model analysis)')
    
    args = parser.parse_args()
    
    # This loader is dedicated to task_allo scenario only
    scenario = 'task_allo'
    exp_name = args.exp_name
    model_name = args.model
    
    if args.result_path:
        # Extract information from path
        # Example: result/experiment_name_gpt-5.1/task_allocation_level_1_with_tools
        path_parts = args.result_path.strip('/').split('/')
        
        # Extract exp_name and model if not specified
        if exp_name is None or model_name is None:
            for part in path_parts:
                if part.startswith('exp_') or '_' in part:
                    # Try to extract model name from folder name
                    # Format: exp_name_model_name
                    if model_name is None:
                        # Try to find model name patterns
                        for known_model in ['gpt-5.1', 'deepseek-v3.2', 'glm-4.7', 'gpt-4.1-mini', 'qwen3-next-80b', 'qwen3-32b', 'qwen3-30b']:
                            if known_model in part:
                                # Extract the full model name after exp_name
                                parts_split = part.split('_')
                                # Find where model name starts
                                for i, p in enumerate(parts_split):
                                    if known_model in p:
                                        model_name = '_'.join(parts_split[i:])
                                        print(f"Auto-detected model: {model_name}")
                                        break
                                break
                    
                    if exp_name is None:
                        # Extract experiment name (before model name)
                        if model_name:
                            exp_name = part.replace(f'_{model_name}', '').replace(f'{model_name}', '')
                        else:
                            exp_name = part
                        print(f"Auto-detected exp_name: {exp_name}")
        
        # Task_allo specific defaults
        game_folder = 'test_games'
        levels = [1]
        
        # Determine category from path
        categories = []
        if 'with_tools' in args.result_path:
            categories.append('with_tools')
        if 'no_tools' in args.result_path:
            categories.append('no_tools')
        if not categories:
            categories = ['with_tools', 'no_tools']
        
        # Single model mode
        if model_name:
            models = [model_name]
        else:
            # Default models
            models = [
                'gpt-5.1',
                'deepseek-v3.2',
                'glm-4.7',
                'gpt-4.1-mini',
                'qwen3-next-80b',
                'qwen3-32b',
                'qwen3-30b',
            ]
        
        print(f"\nLoading task_allo results with auto-detected parameters:")
        print(f"  Experiment: {exp_name}")
        print(f"  Models: {models}")
        print(f"  Levels: {levels}")
        print(f"  Categories: {categories}")
        print(f"  Game folder: {game_folder}")
    
    else:
        # Default: Task_allo scenario
        exp_name = args.exp_name or 'experiment_name'
        models = [
            'gpt-5.1',
            'deepseek-v3.2',
            'glm-4.7',
            'gpt-4.1-mini',
            'qwen3-next-80b',
            'qwen3-32b',
            'qwen3-30b',
        ]
        
        game_folder = 'test_games'
        levels = [1]
        categories = ['with_tools', 'no_tools']
        
        print(f"Using default parameters for task_allo scenario")
    
    # Load all results
    print("\nLoading all results...")
    df = load_all_results_to_dataframe(
        exp_name=exp_name,
        models=models,
        levels=levels,
        start_game_id=args.start,
        end_game_id=args.end,
        base_dir='result',
        game_folder=game_folder,
        n_games=60,
        categories=categories,
        oracle_exp_name=None,  # Set to '{exp_name}_oracle' to load oracle results
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
        if args.result_path:
            # Save in the result directory
            result_dir = os.path.dirname(args.result_path) if os.path.isfile(args.result_path) else args.result_path
            output_file = os.path.join(result_dir, f'results_{args.start}_{args.end}.csv')
        else:
            output_file = os.path.join('result', f'results_task_allo_{args.start}_{args.end}.csv')
        df.to_csv(output_file, index=False)
        print(f"\nDataFrame saved to: {output_file}")
