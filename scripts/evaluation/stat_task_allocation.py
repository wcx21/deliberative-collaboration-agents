"""
Comprehensive Statistics Script for Task Allocation Experiments

Supports:
- Task allocation scenario
- Oracle baseline comparison
- Statistical significance testing (t-test and Wilcoxon)
- Standard error calculation
- Multi-model comparison
- Extensible framework for additional metrics

This script follows the same structure and logic as stat_cooking.py but is
specifically designed for task allocation experiments.
"""

import pickle as pkl
import os
import sys
import numpy as np
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # Create a dummy stats module for basic testing
    class DummyStats:
        @staticmethod
        def ttest_rel(*args, **kwargs):
            raise ImportError("scipy.stats not available")
        @staticmethod
        def ttest_ind(*args, **kwargs):
            raise ImportError("scipy.stats not available")
        @staticmethod
        def wilcoxon(*args, **kwargs):
            raise ImportError("scipy.stats not available")
        @staticmethod
        def mannwhitneyu(*args, **kwargs):
            raise ImportError("scipy.stats not available")
    stats = DummyStats()
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Set up paths relative to this file
from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)
os.chdir(root_dir)
print(os.getcwd())

# Import game loading utilities for task allocation
try:
    from delib_collab.data_generation.task_allocation.io import load_games
except ImportError:
    print("Warning: Could not import load_games from delib_collab.data_generation.task_allocation.io")


# ============================================================================
# Core Result Loading Functions
# ============================================================================

def load_results_task_allo(result_path: str, games: Optional[List] = None,
                            start_game_id: int = 0, end_game_id: int = 60) -> Dict[str, List]:
    """
    Load results for task_allo scenario.
    
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
            # Fallback: read from scores (new format: (agent_score, max_reward, score_ratio, nr, nar))
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
                    # Calculate NR and NAR from score_ratio (assuming they're equal for old format)
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
            import traceback
            traceback.print_exc()
            continue
    
    return {
        'nr': nrs,
        'nar': nars,  # NAR (Normalized Adjusted Reward)
        'vr': vrs,
        'n_tasks': n_tasks,
        'game_ids': game_ids
    }


def load_results_oracle_task_allo(oracle_path: str, format: str = 'pkl') -> Dict[str, List]:
    """
    Load oracle baseline results for task_allo scenario.
    
    Args:
        oracle_path: Path to oracle.pkl file or oracle result directory
        format: 'pkl' for single file, 'dir' for directory structure
        
    Returns:
        Dictionary with same format as load_results_task_allo
    """
    if format == 'pkl':
        # Single oracle.pkl file format
        if not os.path.exists(oracle_path):
            return {'nr': [], 'nar': [], 'vr': [], 'n_tasks': [], 'game_ids': []}
        
        try:
            with open(oracle_path, 'rb') as f:
                oracle_result = pkl.load(f)
            
            nrs, nars, vrs, n_tasks, game_ids = [], [], [], [], []
            
            for key, result in oracle_result.items():
                if not key.startswith('game_'):
                    continue
                
                game_id = int(key.split('_')[-1])
                final_allocation = result['final_proposal']
                
                # Read from evaluation_metrics or scores
                if 'evaluation_metrics' in result:
                    metrics = result['evaluation_metrics']
                    nr = metrics.get('nr', 0.0)
                    nar = metrics.get('nar', 0.0)
                else:
                    scores = result.get('scores', ())
                    if len(scores) >= 5:
                        nr = scores[-2]
                        nar = scores[-1]
                    else:
                        nr = 0.0
                        nar = 0.0
                
                vr = 1 if nr == nar and not (nr == 0 and nar == 0) else 0
                
                nrs.append(nr)
                nars.append(nar)
                vrs.append(vr)
                n_tasks.append(len(final_allocation))
                game_ids.append(game_id)
            
            # Sort by game_id
            sorted_indices = np.argsort(game_ids)
            return {
                'nr': [nrs[i] for i in sorted_indices],
                'nar': [nars[i] for i in sorted_indices],
                'vr': [vrs[i] for i in sorted_indices],
                'n_tasks': [n_tasks[i] for i in sorted_indices],
                'game_ids': [game_ids[i] for i in sorted_indices]
            }
            
        except Exception as e:
            print(f"Error loading oracle.pkl: {e}")
            return {'nr': [], 'nar': [], 'vr': [], 'n_tasks': [], 'game_ids': []}
    
    else:
        # Directory structure format (same as regular results)
        return load_results_task_allo(oracle_path)


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def calculate_statistics(values: List[float], name: str = 'Metric') -> Tuple[str, Dict]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        name: Name of the metric
        
    Returns:
        Tuple of (formatted_string, statistics_dict)
    """
    if len(values) == 0:
        return f"{name}: No data", {'mean': 0, 'std': 0, 'se': 0, 'count': 0}
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1)  # Sample standard deviation
    se = std / np.sqrt(len(values))  # Standard error
    count = len(values)
    
    formatted = f"{name}: {mean:.4f} ± {se:.4f} (n={count})"
    stats_dict = {
        'mean': mean,
        'std': std,
        'se': se,
        'count': count,
        'values': values
    }
    
    return formatted, stats_dict


def test_significance(values1: List[float], values2: List[float], 
                     method: str = 'both', paired: bool = True) -> Dict:
    """
    Perform statistical significance tests between two groups.
    
    Args:
        values1: First group of values
        values2: Second group of values
        method: 'ttest', 'wilcoxon', or 'both'
        paired: Whether data is paired (same games)
        
    Returns:
        Dictionary with p-values and significance indicators
    """
    v1 = np.array(values1)
    v2 = np.array(values2)
    
    # Align data if paired
    if paired and len(v1) != len(v2):
        # Find common game_ids if available
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
    
    if len(v1) == 0 or len(v2) == 0:
        return {'ttest_p': None, 'wilcoxon_p': None, 'significant_ttest': False, 'significant_wilcoxon': False}
    
    results = {}
    
    # t-test
    if method in ['ttest', 'both']:
        if paired:
            t_stat, t_p = stats.ttest_rel(v1, v2)
        else:
            t_stat, t_p = stats.ttest_ind(v1, v2)
        results['ttest_p'] = t_p
        results['significant_ttest'] = t_p < 0.05
    else:
        results['ttest_p'] = None
        results['significant_ttest'] = False
    
    # Wilcoxon signed-rank test
    if method in ['wilcoxon', 'both']:
        if paired:
            try:
                w_stat, w_p = stats.wilcoxon(v1, v2)
            except ValueError:
                w_p = 1.0  # All values are equal
        else:
            w_stat, w_p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        results['wilcoxon_p'] = w_p
        results['significant_wilcoxon'] = w_p < 0.05
    else:
        results['wilcoxon_p'] = None
        results['significant_wilcoxon'] = False
    
    return results


def compare_with_best(all_results_dict: Dict[str, Dict], metric_key: str = 'nar') -> Dict:
    """
    Compare all models with the best performing model.
    
    Args:
        all_results_dict: Dictionary mapping model_name -> stats_dict (from compute_statistics)
        metric_key: Metric to compare (e.g., 'nar', 'nr', 'vr')
        
    Returns:
        Dictionary with comparison results
    """
    # Extract values from stats_dict structure
    # stats_dict structure: {metric_key: {'mean': ..., 'values': [...], ...}}
    def get_values(results, key):
        """Extract values list from stats_dict or raw results dict."""
        if key not in results:
            return None
        metric_data = results[key]
        # Check if it's a stats_dict (has 'values' key) or raw list
        if isinstance(metric_data, dict) and 'values' in metric_data:
            return metric_data['values']
        elif isinstance(metric_data, list):
            return metric_data
        else:
            return None
    
    # Find best model
    best_model = None
    best_mean = -1
    
    model_means = {}
    for model_name, results in all_results_dict.items():
        values = get_values(results, metric_key)
        if values is not None and len(values) > 0:
            mean = np.mean(values)
            model_means[model_name] = mean
            if mean > best_mean:
                best_mean = mean
                best_model = model_name
    
    if best_model is None:
        return {'best_model': None, 'comparisons': {}}
    
    # Compare all others with best
    comparisons = {}
    best_values = get_values(all_results_dict[best_model], metric_key)
    
    if best_values is None or len(best_values) == 0:
        return {'best_model': None, 'comparisons': {}}
    
    for model_name, results in all_results_dict.items():
        model_values = get_values(results, metric_key)
        
        if model_name == best_model:
            comparisons[model_name] = {
                'mean': best_mean,
                'is_best': True,
                'diff': 0,
                'p_ttest': None,
                'p_wilcoxon': None,
                'significant': False
            }
        elif model_values is not None and len(model_values) > 0:
            sig_results = test_significance(best_values, model_values, method='both', paired=True)
            
            mean = np.mean(model_values)
            diff = best_mean - mean
            
            comparisons[model_name] = {
                'mean': mean,
                'is_best': False,
                'diff': diff,
                'p_ttest': sig_results['ttest_p'],
                'p_wilcoxon': sig_results['wilcoxon_p'],
                'significant': sig_results['significant_ttest'] or sig_results['significant_wilcoxon']
            }
    
    return {
        'best_model': best_model,
        'best_mean': best_mean,
        'metric': metric_key,
        'comparisons': comparisons
    }


# ============================================================================
# Main Statistics Computation
# ============================================================================

def compute_statistics(result_path: str, 
                      games: Optional[List] = None, 
                      oracle_path: Optional[str] = None,
                      start_game_id: int = 0, end_game_id: int = 60) -> Dict:
    """
    Compute comprehensive statistics for a task_allo result directory.
    
    Args:
        result_path: Path to result directory
        games: Optional list of game objects
        oracle_path: Optional path to oracle baseline
        
    Returns:
        Dictionary with all computed statistics
    """
    # Load results
    results = load_results_task_allo(result_path, games, start_game_id, end_game_id)
    
    # Calculate statistics for each metric
    stats_dict = {}
    metrics = ['nr', 'nar', 'vr', 'n_tasks']
    
    for metric in metrics:
        if metric in results and len(results[metric]) > 0:
            _, stat_info = calculate_statistics(results[metric], metric)
            stats_dict[metric] = stat_info
        else:
            stats_dict[metric] = {'mean': 0, 'std': 0, 'se': 0, 'count': 0, 'values': []}
    
    # Compute extended metrics if games are provided
    if games:
        try:
            # Filter games by range
            filtered_games = games[start_game_id:end_game_id] if games else None
            if filtered_games:
                extended_metrics = compute_extended_metrics_task_allo(result_path, filtered_games, start_game_id)
                stats_dict['extended_metrics'] = extended_metrics
            else:
                stats_dict['extended_metrics'] = {}
        except Exception as e:
            print(f"Warning: Failed to compute extended metrics: {e}")
            stats_dict['extended_metrics'] = {}
    
    # Load oracle if provided
    oracle_stats = None
    if oracle_path and os.path.exists(oracle_path):
        oracle_results = load_results_oracle_task_allo(oracle_path)
        if len(oracle_results.get('nar', [])) > 0:
            oracle_stats = {}
            for metric in ['nr', 'nar', 'vr']:
                if metric in oracle_results:
                    _, stat_info = calculate_statistics(oracle_results[metric], f'oracle_{metric}')
                    oracle_stats[metric] = stat_info
            
            # Compare with oracle (using NAR as main metric)
            if 'nar' in results and len(results['nar']) > 0:
                oracle_comparison = test_significance(
                    oracle_results['nar'], 
                    results['nar'], 
                    method='both', 
                    paired=True
                )
                stats_dict['oracle_comparison'] = oracle_comparison
    
    stats_dict['oracle_stats'] = oracle_stats
    stats_dict['raw_results'] = results
    
    return stats_dict


# ============================================================================
# Output Formatting Functions
# ============================================================================

def format_latex_table(results_dict: Dict[str, Dict], models: List[str], 
                      metric: str = 'nar', compare_best: bool = True) -> str:
    """
    Generate LaTeX formatted table output.
    
    Args:
        results_dict: Dictionary mapping model_name -> stats_dict
        models: List of model names in order
        metric: Metric to display
        compare_best: Whether to mark significance compared to best
        
    Returns:
        LaTeX formatted string
    """
    if compare_best:
        comparison = compare_with_best(results_dict, metric)
        best_model = comparison['best_model']
    else:
        best_model = None
    
    lines = []
    for model in models:
        if model not in results_dict:
            continue
        
        stats = results_dict[model].get(metric, {})
        mean = stats.get('mean', 0)
        se = stats.get('se', 0)
        
        # Format: mean ± se (values are already in percentage 0-100)
        formatted = f"${mean:.1f}\\pm{se:.1f}$"
        
        # Add significance marker
        if compare_best and best_model and model != best_model:
            comp = comparison['comparisons'].get(model, {})
            if comp.get('significant', False):
                formatted += "^*"
        
        lines.append(formatted)
    
    return " & ".join(lines)


def format_summary(results_dict: Dict, 
                 model_name: str = '', compare_best: bool = False,
                 all_results_dict: Optional[Dict[str, Dict]] = None) -> str:
    """
    Format human-readable summary of statistics as string.
    
    Args:
        results_dict: Statistics dictionary from compute_statistics
        model_name: Name of the model
        compare_best: Whether to compare with best model
        all_results_dict: Dictionary of all models' results for comparison
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*60}")
    if model_name:
        lines.append(f"Model: {model_name}")
    lines.append(f"Scenario: task_allo")
    lines.append(f"{'='*60}")
    
    metrics = ['nr', 'nar', 'vr', 'n_tasks']
    metric_names = {
        'nr': 'Normalized Reward',
        'nar': 'Normalized Adjusted Reward',
        'vr': 'VR (Valid Ratio)',
        'n_tasks': 'Number of Tasks'
    }
    
    for metric in metrics:
        if metric in results_dict:
            stats = results_dict[metric]
            count = stats.get('count', 0)
            if count > 0:
                formatted, _ = calculate_statistics(stats.get('values', []), metric_names.get(metric, metric))
                lines.append(formatted)
    
    # Print extended metrics if available
    if 'extended_metrics' in results_dict and results_dict['extended_metrics']:
        lines.append(f"\nExtended Metrics:")
        ext_metrics = results_dict['extended_metrics']
        
        if 'state_estimation_accuracy' in ext_metrics:
            stats = ext_metrics['state_estimation_accuracy']
            if stats.get('count', 0) > 0:
                formatted, _ = calculate_statistics(stats.get('values', []), 'State Estimation Accuracy')
                lines.append(formatted)
        
        if 'value_estimation_accuracy' in ext_metrics:
            stats = ext_metrics['value_estimation_accuracy']
            if stats.get('count', 0) > 0:
                formatted, _ = calculate_statistics(stats.get('values', []), 'Value Estimation Accuracy')
                lines.append(formatted)
        
        if 'hallucination_rate' in ext_metrics:
            stats = ext_metrics['hallucination_rate']
            if stats.get('count', 0) > 0:
                formatted, _ = calculate_statistics(stats.get('values', []), 'Hallucination Rate')
                lines.append(formatted)
    
    # Print oracle comparison if available
    if results_dict.get('oracle_stats'):
        lines.append(f"\nOracle Baseline:")
        oracle_stats = results_dict['oracle_stats']
        for metric in ['nr', 'nar', 'vr']:
            if metric in oracle_stats:
                stats = oracle_stats[metric]
                formatted, _ = calculate_statistics(stats.get('values', []), f'Oracle {metric_names.get(metric, metric)}')
                lines.append(formatted)
        
        if 'oracle_comparison' in results_dict:
            comp = results_dict['oracle_comparison']
            lines.append(f"\nComparison with Oracle (nar):")
            lines.append(f"  t-test p-value: {comp.get('ttest_p', 'N/A'):.4f}")
            lines.append(f"  Wilcoxon p-value: {comp.get('wilcoxon_p', 'N/A'):.4f}")
            lines.append(f"  Significant: {comp.get('significant_ttest', False) or comp.get('significant_wilcoxon', False)}")
    
    # Print best model comparison if requested
    if compare_best and all_results_dict and len(all_results_dict) > 1:
        lines.append(f"\n{'='*60}")
        lines.append("Comparison with Best Model:")
        lines.append(f"{'='*60}")
        
        comparison_metrics = ['nar', 'nr', 'vr']
        
        for metric in comparison_metrics:
            if metric not in metrics:
                continue
            
            comparison = compare_with_best(all_results_dict, metric)
            best_model = comparison['best_model']
            if best_model:
                lines.append(f"\nBest model for {metric_names.get(metric, metric)}: {best_model} ({comparison['best_mean']:.4f})")
                lines.append(f"Comparisons:")
                for model, comp in comparison['comparisons'].items():
                    if not comp['is_best']:
                        sig_marker = " *" if comp['significant'] else ""
                        lines.append(f"  {model}: {comp['mean']:.4f} (diff: {comp['diff']:.4f}, "
                              f"p_ttest={comp['p_ttest']:.4f}, p_wilcoxon={comp['p_wilcoxon']:.4f}){sig_marker}")
    
    return '\n'.join(lines)


def print_summary(results_dict: Dict, 
                 model_name: str = '', compare_best: bool = False,
                 all_results_dict: Optional[Dict[str, Dict]] = None) -> None:
    """Print summary to console (wrapper around format_summary)."""
    print(format_summary(results_dict, model_name, compare_best, all_results_dict))


# ============================================================================
# Extended Metrics (State Estimation, Value Estimation, Hallucination Rate)
# ============================================================================

def load_estimation_from_full_record_task_allo(result_path: str, game_id: int) -> Optional[Dict]:
    """
    Load estimation data from full_record.pkl for task allocation scenario.
    
    Args:
        result_path: Path to result directory containing game_* folders
        game_id: Game ID
        
    Returns:
        Dictionary with 'estimated_total_resources' and 'estimated_overall_preferences',
        or None if not found
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
        
        # Find the last round
        round_keys = [k for k in record.keys() if k.startswith('round ')]
        if not round_keys:
            return None
        
        # Sort rounds and get the last one
        last_round = sorted(round_keys, key=lambda x: int(x.split(' ')[1]))[-1]
        
        # Get the last agent's estimation (typically agent_2/Leader)
        agent_keys = [k for k in record[last_round].keys() if k.startswith('agent_')]
        if not agent_keys:
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
                            'estimated_overall_preferences': overall_preferences
                        }
        
        return None
        
    except Exception as e:
        print(f"Error loading full_record for game {game_id}: {e}")
        return None


def compute_extended_metrics_task_allo(result_path: str, games: List, start_game_id: int = 0) -> Dict:
    """
    Compute extended metrics for task allocation scenario from full_record.
    
    Metrics computed:
    - state_estimation_accuracy: State Estimation Accuracy (obs_er_rt)
    - value_estimation_accuracy: Value Estimation Accuracy (useful_pref_sim)
    - hallucination_rate: Hallucination Rate (hall_rt)
    
    Args:
        result_path: Path to result directory
        games: List of game objects
        
    Returns:
        Dictionary with extended metrics
    """
    from delib_collab.data_generation.task_allocation.planner import vectorize_task_allocation_state
    
    state_acc_ls = []
    value_acc_ls = []
    hall_rt_ls = []
    valid_ids = []
    
    for idx, game in enumerate(games):
        game_id = start_game_id + idx
        estimation_data = load_estimation_from_full_record_task_allo(result_path, game_id)
        
        if not estimation_data:
            continue
        
        try:
            # Get ground truth resource state
            gt_resource_state = game.game_resource_state
            gt_vec_state = game.vec_resource_state
            
            # Get best allocation and its resource requirements
            best_allocation = game.best_allocation
            constraint_matrix = game.constraint_matrix
            
            # Calculate resource requirements for best allocation
            if best_allocation is not None and np.sum(best_allocation) > 0:
                best_allocation_resource_usage = constraint_matrix @ best_allocation
            else:
                best_allocation_resource_usage = np.zeros_like(gt_vec_state)
            
            # Process estimated resources
            estimated_resources = estimation_data['estimated_total_resources']
            # Handle both dict and nested formats
            if isinstance(estimated_resources, dict):
                agent_private_resources = estimated_resources.get('agent_private_resources', {})
                public_resources = estimated_resources.get('public_resources', {})
            else:
                continue
            
            # Vectorize estimated resources
            estimated_vec_state = vectorize_task_allocation_state(
                agent_private_resources,
                public_resources,
                game.private_resource_list,
                game.public_resource_list
            )
            
            # Calculate State Estimation Accuracy
            # Similar to cook scenario: calculate lack of essential resources
            essential_diff = (gt_vec_state[np.where(best_allocation_resource_usage > 0)] -
                            estimated_vec_state[np.where(best_allocation_resource_usage > 0)])
            lack_resources = np.clip(essential_diff, 0, 1e5).sum()
            
            if best_allocation_resource_usage.sum() > 0:
                state_acc = 1 - (lack_resources) / (best_allocation_resource_usage.sum())
            else:
                state_acc = 0.0
            
            # Calculate Hallucination Rate (false resources: estimated > 0 where gt == 0)
            false_resources = estimated_vec_state[np.where(gt_vec_state == 0)].sum()
            if gt_vec_state.sum() > 0:
                hall_rt = false_resources / (gt_vec_state.sum())
            else:
                hall_rt = 0.0
            
            # Process estimated preferences
            estimated_preferences = estimation_data['estimated_overall_preferences']
            # Handle both dict and nested formats
            if isinstance(estimated_preferences, dict):
                # Convert to value matrix format
                # estimated_preferences format: {'agent_0': {'task_1': efficiency, ...}, ...}
                num_agents = len(game.agents)
                num_tasks = len(game.tasks)
                estimated_value_matrix = np.zeros((num_agents, num_tasks))
                
                for agent_idx, agent_name in enumerate(game.agents):
                    agent_prefs = estimated_preferences.get(agent_name, {})
                    for task_idx, task_name in enumerate(game.tasks):
                        estimated_value_matrix[agent_idx, task_idx] = agent_prefs.get(task_name, 0)
            else:
                continue
            
            # Get ground truth value matrix
            gt_value_matrix = game.value_matrix
            
            # Calculate Value Estimation Accuracy (cosine similarity on value matrices)
            # Flatten both matrices for comparison
            gt_values_flat = gt_value_matrix.flatten()
            est_values_flat = estimated_value_matrix.flatten()
            
            if np.linalg.norm(est_values_flat) == 0:
                value_acc = 0
            else:
                dp = np.dot(est_values_flat, gt_values_flat)
                value_acc = dp / np.linalg.norm(est_values_flat) / np.linalg.norm(gt_values_flat)
            
            state_acc_ls.append(state_acc)
            value_acc_ls.append(value_acc)
            hall_rt_ls.append(hall_rt)
            valid_ids.append(game_id)
            
        except Exception as e:
            print(f"Error computing extended metrics for game {game_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return {
        'state_estimation_accuracy': {
            'values': state_acc_ls,
            'mean': np.mean(state_acc_ls) if state_acc_ls else 0.0,
            'std': np.std(state_acc_ls, ddof=1) if len(state_acc_ls) > 1 else 0.0,
            'se': np.std(state_acc_ls, ddof=1) / np.sqrt(len(state_acc_ls)) if state_acc_ls else 0.0,
            'count': len(state_acc_ls)
        },
        'value_estimation_accuracy': {
            'values': value_acc_ls,
            'mean': np.mean(value_acc_ls) if value_acc_ls else 0.0,
            'std': np.std(value_acc_ls, ddof=1) if len(value_acc_ls) > 1 else 0.0,
            'se': np.std(value_acc_ls, ddof=1) / np.sqrt(len(value_acc_ls)) if value_acc_ls else 0.0,
            'count': len(value_acc_ls)
        },
        'hallucination_rate': {
            'values': hall_rt_ls,
            'mean': np.mean(hall_rt_ls) if hall_rt_ls else 0.0,
            'std': np.std(hall_rt_ls, ddof=1) if len(hall_rt_ls) > 1 else 0.0,
            'se': np.std(hall_rt_ls, ddof=1) / np.sqrt(len(hall_rt_ls)) if hall_rt_ls else 0.0,
            'count': len(hall_rt_ls)
        },
        'valid_ids': valid_ids
    }


def compute_extended_metrics(result_path: str, games: List, log_path: Optional[str] = None) -> Dict:
    """
    Compute extended metrics for task allocation scenario.
    
    Args:
        result_path: Path to result directory
        games: List of game objects
        log_path: Optional path to log directory (not used, kept for compatibility)
        
    Returns:
        Dictionary with extended metrics
    """
    return compute_extended_metrics_task_allo(result_path, games)


# ============================================================================
# Main Execution
# ============================================================================

def run_statistics(exp_name: str, models: List[str], levels: List[int],
                  base_dir: str = 'result',
                  game_folder: str = 'task_allocation_games', n_games: int = 60,
                  categories: Optional[List[str]] = None,
                  oracle_path: Optional[str] = None,
                  compare_best: bool = True,
                  output_file: Optional[str] = None,
                  start_game_id: int = 0, end_game_id: int = 60) -> None:
    """
    Run comprehensive statistics for multiple models and levels.
    
    Args:
        exp_name: Experiment name
        models: List of model names
        levels: List of levels to process (for task_allo, typically [1])
        base_dir: Base directory for results (default: 'result')
        game_folder: Game data folder name
        n_games: Number of games
        categories: List of categories (e.g., ['with_tools', 'no_tools']). If None, uses both.
        oracle_path: Optional path to oracle baseline
        compare_best: Whether to compare with best model
        output_file: Optional path to output file. If None, generates default name in result directory.
        start_game_id: Start game ID (inclusive, default: 0)
        end_game_id: End game ID (exclusive, default: 60)
    """
    if categories is None:
        categories = ['with_tools', 'no_tools']
    
    # Convert base_dir to absolute path if it's relative
    # This ensures the path works regardless of where the script is run from
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(root_dir, base_dir)
    base_dir = os.path.normpath(base_dir)
    
    # Convert oracle_path to absolute path if it's relative and not None
    if oracle_path and not os.path.isabs(oracle_path):
        oracle_path = os.path.join(root_dir, oracle_path)
        oracle_path = os.path.normpath(oracle_path)
    
    # Prepare output file
    if output_file is None:
        # Generate default output file name (without timestamp, will overwrite on rerun)
        output_file = os.path.join(base_dir, f"{exp_name}_statistics.txt")
    else:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else base_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Open file for writing
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"Statistics Report: {exp_name}")
    report_lines.append(f"Scenario: task_allo")
    report_lines.append(f"Models: {', '.join(models)}")
    report_lines.append(f"Levels: {', '.join(map(str, levels))}")
    report_lines.append(f"Categories: {', '.join(categories)}")
    report_lines.append(f"Game ID Range: [{start_game_id}, {end_game_id})")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    
    all_results = {}  # For best model comparison
    
    for level in levels:
        level_header = f"\n{'#'*60}\nLevel {level}\n{'#'*60}"
        print(level_header)
        report_lines.append(level_header)
        
        # Load games (only load the range we need)
        games = None
        try:
            all_games = load_games(target_path=game_folder, game_batch_name='level_1_and_2', n_games=n_games)
            games = all_games[start_game_id:end_game_id] if all_games else None
        except Exception as e:
            error_msg = f"Warning: Could not load games: {e}"
            print(error_msg)
            report_lines.append(error_msg)
            games = None
        
        for category in categories:
            category_header = f"\n{'-'*60}\nCategory: {category}\n{'-'*60}"
            print(category_header)
            report_lines.append(category_header)
            
            for model in models:
                folder_name = f"{exp_name}_{model}"
                # For oracle, use baseline subdirectory
                if 'oracle' in exp_name.lower():
                    category_path = f"task_allocation_level_{level}_baseline_{category}"
                else:
                    category_path = f"task_allocation_level_{level}_{category}"
                result_path = os.path.join(base_dir, folder_name, category_path)
                
                if not os.path.exists(result_path):
                    error_msg = f"  {model}: Path not found: {result_path}"
                    print(error_msg)
                    report_lines.append(error_msg)
                    continue
                
                # Compute statistics
                stats_dict = compute_statistics(
                    result_path=result_path,
                    games=games,  # Pass games for extended metrics computation
                    oracle_path=oracle_path,
                    start_game_id=start_game_id,
                    end_game_id=end_game_id
                )
                
                # Store for comparison
                if model not in all_results:
                    all_results[model] = {}
                all_results[model][f"level_{level}_{category}"] = stats_dict
                
                # Print and save summary
                summary_text = format_summary(
                    results_dict=stats_dict,
                    model_name=model,
                    compare_best=False,  # Will do batch comparison later
                    all_results_dict=None
                )
                print(summary_text)
                report_lines.append(summary_text)
            
            # Print best model comparison for this category
            if compare_best and len(models) > 1:
                category_results = {}
                for model in models:
                    key = f"level_{level}_{category}"
                    if model in all_results and key in all_results[model]:
                        category_results[model] = all_results[model][key]
                
                if len(category_results) > 1:
                    comparison_header = f"\n{'='*60}\nBest Model Comparison - Level {level}, {category}\n{'='*60}"
                    print(comparison_header)
                    report_lines.append(comparison_header)
                    
                    # Compare for main metric (NAR)
                    # main_metric = 'nar'
                    main_metric = 'nr'
                    comparison = compare_with_best(category_results, main_metric)
                    
                    if comparison['best_model']:
                        best_msg = f"Best model: {comparison['best_model']} ({comparison['best_mean']:.4f})"
                        print(best_msg)
                        report_lines.append(best_msg)
                        
                        models_msg = "\nAll models:"
                        print(models_msg)
                        report_lines.append(models_msg)
                        for model in models:
                            if model in comparison['comparisons']:
                                comp = comparison['comparisons'][model]
                                sig_marker = " *" if comp['significant'] and not comp['is_best'] else ""
                                model_msg = f"  {model}: {comp['mean']:.4f}{sig_marker}"
                                print(model_msg)
                                report_lines.append(model_msg)
    
    # Write report to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n{'='*60}")
        print(f"Report saved to: {output_file}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\nWarning: Failed to write report file: {e}")


def test_statistics(result_path: str, verbose: bool = True) -> Dict:
    """
    Test the correctness of the statistics code.
    
    Args:
        result_path: Path to result folder (e.g. 'result/debug_test_260103_gemini/task_allocation_level_1_no_tools')
        verbose: Whether to output detailed information
        
    Returns:
        Statistics result dictionary
    """
    print('='*70)
    print('Statistics Code Test')
    print('='*70)
    print(f'\nResult path: {result_path}')
    print(f'Path exists: {os.path.exists(result_path)}')
    
    if not os.path.exists(result_path):
        print(f'\nError: path does not exist: {result_path}')
        return {}
    
    print('\n' + '-'*70)
    print('Step 1: Load result files')
    print('-'*70)
    results = load_results_task_allo(result_path)
    
    print(f'  Successfully loaded {len(results["nr"])} games')
    print(f'  Game IDs: {results["game_ids"]}')
    
    if verbose:
        print(f'\n  Detailed data:')
        print(f'    NR values: {[round(n, 2) for n in results["nr"]]}')
        print(f'    NAR values: {[round(n, 2) for n in results["nar"]]}')
        print(f'    VR values: {results["vr"]}')
        print(f'    n_tasks: {results["n_tasks"]}')
    
    if len(results["nr"]) == 0:
        print('\nError: no results loaded')
        return {}
    
    print('\n' + '-'*70)
    print('Step 2: Compute statistics')
    print('-'*70)
    stats_dict = compute_statistics(result_path)
    
    print('\n' + '-'*70)
    print('Step 3: Statistics results')
    print('-'*70)
    
    metrics = ['nr', 'nar', 'vr', 'n_tasks']
    metric_names = {
        'nr': 'Normalized Reward',
        'nar': 'Normalized Adjusted Reward',
        'vr': 'VR (Valid Ratio)',
        'n_tasks': 'Number of Tasks'
    }
    
    for metric in metrics:
        if metric in stats_dict:
            stats = stats_dict[metric]
            count = stats.get('count', 0)
            if count > 0:
                mean = stats.get('mean', 0)
                std = stats.get('std', 0)
                se = stats.get('se', 0)
                values = stats.get('values', [])
                
                print(f'\n{metric_names.get(metric, metric)}:')
                print(f'  Mean: {mean:.4f}')
                print(f'  Std:  {std:.4f}')
                print(f'  SE:   {se:.4f}')
                print(f'  Count: {count}')
                if verbose and len(values) <= 20:
                    print(f'  Values: {[round(v, 2) for v in values]}')
    
    print('\n' + '-'*70)
    print('Step 4: Manual verification')
    print('-'*70)
    
    nr_values = np.array(results['nr'])
    nar_values = np.array(results['nar'])
    vr_values = np.array(results['vr'])
    
    manual_nr_mean = np.mean(nr_values)
    manual_nr_std = np.std(nr_values, ddof=1)
    manual_nr_se = manual_nr_std / np.sqrt(len(nr_values))
    
    manual_nar_mean = np.mean(nar_values)
    manual_nar_std = np.std(nar_values, ddof=1)
    manual_nar_se = manual_nar_std / np.sqrt(len(nar_values))
    
    manual_vr_mean = np.mean(vr_values)
    
    nr_stats = stats_dict.get('nr', {})
    nar_stats = stats_dict.get('nar', {})
    vr_stats = stats_dict.get('vr', {})
    
    print(f'\nNR verification:')
    print(f'  Manual:  mean={manual_nr_mean:.6f}, std={manual_nr_std:.6f}, se={manual_nr_se:.6f}')
    print(f'  Script:  mean={nr_stats.get("mean", 0):.6f}, std={nr_stats.get("std", 0):.6f}, se={nr_stats.get("se", 0):.6f}')
    nr_match = (abs(nr_stats.get('mean', 0) - manual_nr_mean) < 0.0001 and
                abs(nr_stats.get('std', 0) - manual_nr_std) < 0.0001 and
                abs(nr_stats.get('se', 0) - manual_nr_se) < 0.0001)
    print(f'  Match: {"PASS" if nr_match else "FAIL"}')
    
    print(f'\nNAR verification:')
    print(f'  Manual:  mean={manual_nar_mean:.6f}, std={manual_nar_std:.6f}, se={manual_nar_se:.6f}')
    print(f'  Script:  mean={nar_stats.get("mean", 0):.6f}, std={nar_stats.get("std", 0):.6f}, se={nar_stats.get("se", 0):.6f}')
    nar_match = (abs(nar_stats.get('mean', 0) - manual_nar_mean) < 0.0001 and
                 abs(nar_stats.get('std', 0) - manual_nar_std) < 0.0001 and
                 abs(nar_stats.get('se', 0) - manual_nar_se) < 0.0001)
    print(f'  Match: {"PASS" if nar_match else "FAIL"}')
    
    print(f'\nVR verification:')
    print(f'  Manual:  mean={manual_vr_mean:.6f}')
    print(f'  Script:  mean={vr_stats.get("mean", 0):.6f}')
    vr_match = abs(vr_stats.get('mean', 0) - manual_vr_mean) < 0.0001
    print(f'  Match: {"PASS" if vr_match else "FAIL"}')
    
    print(f'\nVR logic verification:')
    all_vr_correct = True
    for i, game_id in enumerate(results['game_ids']):
        nr = results['nr'][i]
        nar = results['nar'][i]
        vr = results['vr'][i]
        expected_vr = 1 if nr == nar and not (nr == 0 and nar == 0) else 0
        is_correct = vr == expected_vr
        all_vr_correct = all_vr_correct and is_correct
        if verbose or not is_correct:
            status = 'PASS' if is_correct else 'FAIL'
            print(f'  {status} Game {game_id}: NR={nr:.2f}, NAR={nar:.2f}, VR={vr} (expected={expected_vr})')
    
    print('\n' + '='*70)
    print('Verification Summary')
    print('='*70)
    all_passed = nr_match and nar_match and vr_match and all_vr_correct
    
    print(f'NR computation:  {"PASS" if nr_match else "FAIL"}')
    print(f'NAR computation: {"PASS" if nar_match else "FAIL"}')
    print(f'VR computation:  {"PASS" if vr_match else "FAIL"}')
    print(f'VR logic:        {"PASS" if all_vr_correct else "FAIL"}')
    
    if all_passed:
        print('\nAll verifications passed. Statistics code is correct.')
    else:
        print('\nSome verifications failed. Please check.')
    
    return stats_dict


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test statistics code for task_allo results')
    parser.add_argument('result_path', type=str, nargs='?', 
                       help='Path to result directory (e.g., result/debug_test_260103_gemini/task_allocation_level_1_no_tools)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed information')
    parser.add_argument('--run-full', action='store_true',
                       help='Run full statistics (original behavior)')
    parser.add_argument('-s', '--start', type=int, default=0,
                       help='Start game ID (inclusive, default: 0)')
    parser.add_argument('-e', '--end', type=int, default=60,
                       help='End game ID (exclusive, default: 60)')
    
    args = parser.parse_args()
    
    if args.run_full:
        # Original full statistics run
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
        
        # Run statistics for regular models
        run_statistics(
            exp_name=root_exp_name,
            models=models,
            levels=[1],
            base_dir='result',
            game_folder='task_allocation_games',
            n_games=60,
            compare_best=True,
            start_game_id=args.start,
            end_game_id=args.end
        )
        
        run_statistics(
            exp_name=oracle_exp_name,
            models=models,
            levels=[1],
            base_dir='result',
            game_folder='task_allocation_games',
            n_games=60,
            categories=['with_tools', 'no_tools'],
            compare_best=True,
            start_game_id=args.start,
            end_game_id=args.end
        )
    elif args.result_path:
        test_statistics(args.result_path, verbose=args.verbose)
    else:
        # Default: test with example path
        print("Usage:")
        print("  python stat_task_allocation.py <result_path> [--verbose]")
        print("  python stat_task_allocation.py <result_path> --save [-o output.txt]")
        print("  python stat_task_allocation.py --run-full  # Run full statistics")
        print("  python stat_task_allocation.py --run-full -s 0 -e 30  # Run with game ID range")
        print("\nExample:")
        print("  python stat_task_allocation.py result/exp_name/task_allocation_level_1_no_tools --verbose")
        print("  python stat_task_allocation.py result/exp_name/task_allocation_level_1_no_tools --save")

