"""Task allocation evaluation metrics (NR, NAR, VR)."""

import numpy as np
import json
from typing import Dict, List, Tuple, Any
import os


def calculate_nr(final_reward: float, max_reward: float) -> float:
    """Calculate NR (Normalized Reward) = (reward / max_reward) * 100."""
    if max_reward == 0:
        return 0.0
    nr = (final_reward / max_reward) * 100.0
    return round(nr, 2)


def calculate_nar(
    final_allocation: Dict[str, str],
    game,
    max_reward: float
) -> Tuple[float, Dict[str, str], float]:
    """Calculate NAR (Normalized Adjusted Reward).

    For invalid allocations, greedily selects a feasible subset and applies
    a penalty proportional to the fraction of invalid entries.
    """
    try:
        
        final_reward = game.evaluate_allocation(final_allocation)
        
        if final_reward > 0:
            
            nar = (final_reward / max_reward) * 100.0 if max_reward > 0 else 0.0
            return round(nar, 2), final_allocation, final_reward
        
        
        submenu, submenu_reward = _greedy_submenu_selection(final_allocation, game)
        
        original_size = len(final_allocation)
        submenu_size = len(submenu)
        
        if max_reward == 0 or original_size == 0:
            nar = 0.0
        else:
            nar = (submenu_reward / max_reward) * (submenu_size / original_size) * 100.0
        
        return round(nar, 2), submenu, submenu_reward
    except Exception as e:
        print(f"Error calculating NAR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, {}, 0.0


def _greedy_submenu_selection(
    allocation: Dict[str, str],
    game
) -> Tuple[Dict[str, str], float]:
    """Greedy feasible-subset selection sorted by task value (descending)."""
    if not allocation:
        return {}, 0.0
    
    
    task_values = []
    for task_name, agent_name in allocation.items():
        if task_name in game.tasks and agent_name in game.agents:
            agent_idx = game.agents.index(agent_name)
            task_idx = game.tasks.index(task_name)
            value = game.value_matrix[agent_idx, task_idx]
            task_values.append((task_name, agent_name, value))
    
    
    task_values.sort(key=lambda x: x[2], reverse=True)
    
    submenu = {}
    for task_name, agent_name, value in task_values:
        test_menu = submenu.copy()
        test_menu[task_name] = agent_name
        
        test_reward = game.evaluate_allocation(test_menu)
        if test_reward > 0:
            submenu = test_menu
    
    
    submenu_reward = game.evaluate_allocation(submenu) if submenu else 0.0
    
    return submenu, submenu_reward


def calculate_game_metrics(
    final_allocation: Dict[str, str],
    game,
    max_reward: float
) -> Dict[str, Any]:
    """Calculate all metrics for a single game."""
    final_reward = game.evaluate_allocation(final_allocation)
    is_valid = final_reward > 0
    
    nr = calculate_nr(final_reward, max_reward)
    
    nar, submenu, submenu_reward = calculate_nar(final_allocation, game, max_reward)
    
    metrics = {
        'final_reward': round(final_reward, 2),
        'max_reward': round(max_reward, 2),
        'is_valid': is_valid,
        'nr': nr,
        'nar': nar,
        'submenu': submenu,
        'submenu_reward': round(submenu_reward, 2),
        'original_allocation_size': len(final_allocation),
        'submenu_size': len(submenu)
    }
    
    return metrics


def aggregate_metrics(all_game_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all games (mean, std, and VR)."""
    if not all_game_metrics:
        return {
            'n_games': 0,
            'vr': 0.0,
            'nr_mean': 0.0,
            'nr_std': 0.0,
            'nar_mean': 0.0,
            'nar_std': 0.0
        }
    
    n_games = len(all_game_metrics)
    
    
    nr_list = [m['nr'] for m in all_game_metrics]
    nar_list = [m['nar'] for m in all_game_metrics]
    valid_list = [m['is_valid'] for m in all_game_metrics]
    
    n_valid = sum(valid_list)
    vr = (n_valid / n_games) * 100.0
    
    nr_mean = np.mean(nr_list)
    nr_std = np.std(nr_list, ddof=1) if n_games > 1 else 0.0
    
    nar_mean = np.mean(nar_list)
    nar_std = np.std(nar_list, ddof=1) if n_games > 1 else 0.0
    
    aggregated = {
        'n_games': n_games,
        'n_valid': n_valid,
        'n_invalid': n_games - n_valid,
        'vr': round(vr, 2),
        'nr_mean': round(nr_mean, 2),
        'nr_std': round(nr_std, 2),
        'nar_mean': round(nar_mean, 2),
        'nar_std': round(nar_std, 2),
        'nr_list': [round(x, 2) for x in nr_list],
        'nar_list': [round(x, 2) for x in nar_list],
        'valid_list': valid_list
    }
    
    return aggregated


def save_metrics_summary(
    all_game_metrics: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
    output_path: str
):
    """Save metrics summary to JSON and human-readable TXT files."""
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'summary': aggregated_metrics,
        'per_game_metrics': all_game_metrics
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON metrics file: {e}")
        raise
    
    txt_path = output_path.replace('.json', '.txt')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Task Allocation Metrics Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("[Summary]\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total games:        {aggregated_metrics['n_games']}\n")
            f.write(f"Valid allocations:  {aggregated_metrics['n_valid']}\n")
            f.write(f"Invalid allocations:{aggregated_metrics['n_invalid']}\n")
            f.write(f"\n")
            f.write(f"VR (Valid Ratio):   {aggregated_metrics['vr']:.2f}%\n")
            f.write(f"\n")
            f.write(f"NR (Normalized Reward):\n")
            f.write(f"  Mean:             {aggregated_metrics['nr_mean']:.2f}%\n")
            f.write(f"  Std:              {aggregated_metrics['nr_std']:.2f}%\n")
            f.write(f"\n")
            f.write(f"NAR (Normalized Adjusted Reward):\n")
            f.write(f"  Mean:             {aggregated_metrics['nar_mean']:.2f}%\n")
            f.write(f"  Std:              {aggregated_metrics['nar_std']:.2f}%\n")
            f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("[Per-Game Metrics]\n")
            f.write("=" * 80 + "\n\n")
            
            for i, metrics in enumerate(all_game_metrics):
                f.write(f"Game {i}:\n")
                f.write(f"  Final Reward:     {metrics['final_reward']:.2f}\n")
                f.write(f"  Max Reward:       {metrics['max_reward']:.2f}\n")
                f.write(f"  Is Valid:         {metrics['is_valid']}\n")
                f.write(f"  NR:               {metrics['nr']:.2f}%\n")
                f.write(f"  NAR:              {metrics['nar']:.2f}%\n")
                if not metrics['is_valid']:
                    f.write(f"  Original Size:    {metrics['original_allocation_size']}\n")
                    f.write(f"  Submenu Size:     {metrics['submenu_size']}\n")
                    f.write(f"  Submenu Reward:   {metrics['submenu_reward']:.2f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
    except Exception as e:
        print(f"Error saving TXT metrics file: {e}")
        raise


def print_game_metrics(game_id: int, metrics: Dict[str, Any]):
    """Print metrics for a single game."""
    print(f"\n{'='*60}")
    print(f"Game {game_id} Metrics")
    print(f"{'='*60}")
    print(f"Final Reward:       {metrics['final_reward']:.2f}")
    print(f"Max Reward:         {metrics['max_reward']:.2f}")
    print(f"Is Valid:           {metrics['is_valid']}")
    print(f"NR:                 {metrics['nr']:.2f}%")
    print(f"NAR:                {metrics['nar']:.2f}%")
    
    if not metrics['is_valid']:
        print(f"\n  Invalid allocation, repaired subset:")
        print(f"  Original size:    {metrics['original_allocation_size']}")
        print(f"  Subset size:      {metrics['submenu_size']}")
        print(f"  Subset reward:    {metrics['submenu_reward']:.2f}")
        print(f"  Repaired subset:  {metrics['submenu']}")
    
    print(f"{'='*60}\n")


def print_aggregated_metrics(aggregated: Dict[str, Any]):
    """Print aggregated metrics summary."""
    print(f"\n{'='*80}")
    print(f"Aggregated Metrics Summary")
    print(f"{'='*80}")
    print(f"Total games:        {aggregated['n_games']}")
    print(f"Valid allocations:  {aggregated['n_valid']}")
    print(f"Invalid allocations:{aggregated['n_invalid']}")
    print(f"\n")
    print(f"VR (Valid Ratio):   {aggregated['vr']:.2f}%")
    print(f"\n")
    print(f"NR (Normalized Reward):")
    print(f"  Mean:             {aggregated['nr_mean']:.2f}%")
    print(f"  Std:              {aggregated['nr_std']:.2f}%")
    print(f"\n")
    print(f"NAR (Normalized Adjusted Reward):")
    print(f"  Mean:             {aggregated['nar_mean']:.2f}%")
    print(f"  Std:              {aggregated['nar_std']:.2f}%")
    print(f"{'='*80}\n")


# ==================== Test ====================

def test_metrics():
    """Test metric calculations with mock data."""
    print("Testing metric calculations...")
    
    
    class MockGame:
        def __init__(self):
            self.tasks = ['Task1', 'Task2', 'Task3']
            self.agents = ['agent_0', 'agent_1', 'agent_2']
            self.value_matrix = np.array([
                [10, 8, 12],
                [8, 10, 9],
                [12, 6, 10]
            ])
        
        def evaluate_allocation(self, allocation):
            
            if len(allocation) == len(self.tasks):
                total = 0
                for task, agent in allocation.items():
                    if task in self.tasks and agent in self.agents:
                        task_idx = self.tasks.index(task)
                        agent_idx = self.agents.index(agent)
                        total += self.value_matrix[agent_idx, task_idx]
                return total
            return 0.0
    
    game = MockGame()
    max_reward = 32.0
    
    print("\nTest 1: Valid allocation")
    allocation1 = {'Task1': 'agent_2', 'Task2': 'agent_1', 'Task3': 'agent_0'}
    metrics1 = calculate_game_metrics(allocation1, game, max_reward)
    print_game_metrics(0, metrics1)
    
    print("\nTest 2: Invalid allocation (missing tasks)")
    allocation2 = {'Task1': 'agent_2', 'Task2': 'agent_1'}
    metrics2 = calculate_game_metrics(allocation2, game, max_reward)
    print_game_metrics(1, metrics2)
    
    print("\nTest 3: Aggregated metrics")
    all_metrics = [metrics1, metrics2]
    aggregated = aggregate_metrics(all_metrics)
    print_aggregated_metrics(aggregated)
    
    print("Test completed!")


if __name__ == '__main__':
    test_metrics()

