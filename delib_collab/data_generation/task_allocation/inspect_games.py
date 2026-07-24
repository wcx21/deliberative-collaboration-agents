"""
Game inspection and testing script for task allocation.

Loads game objects, prints game info, validates solver results, and generates
statistical reports. Focuses on Level 1.
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=""):
        if desc:
            print(f"{desc}...")
        return iterable

import numpy as np

from delib_collab.data_generation.task_allocation import io as load_data_util
from delib_collab.data_generation.task_allocation.game import TaskAllocationGame
from delib_collab.data_generation.task_allocation.database import get_all_domains, load_database_domain
from collections import Counter, defaultdict


def detailed_test_level1(game_folder='task_allocation_demo_01', n_games=None, verbose=True):
    """Print detailed info for each Level 1 game."""
    games = load_data_util.load_games(
        target_path=game_folder,
        game_batch_name='level_1_and_2',
        n_games=n_games
    )
    
    print(f"\n{'='*60}")
    print(f"Detailed Test - Level 1 Games")
    print(f"{'='*60}")
    print(f"Game folder: {game_folder}")
    print(f"Num games: {len(games)}")
    print(f"{'='*60}\n")
    
    for i, game in enumerate(games):
        print(f"\n{'='*60}")
        print(f"Game {i}")
        print(f"{'='*60}")
        
        print(f"\nBasic Info:")
        print(f"  Tasks: {game.tasks}")
        print(f"  Resources: {game.resources}")
        print(f"  Agents: {game.agents}")
        
        print(f"\nResource State (Ground Truth):")
        print(f"  Private resources:")
        for agent_name in game.agents:
            agent_display_name = game.agents_config[agent_name].get('name', agent_name)
            private_resources = game.game_resource_state['agent_private_resources'][agent_name]
            print(f"    {agent_display_name}: {private_resources}")
        print(f"  Public resources: {game.game_resource_state['public_resources']}")
        
        print(f"\nAgent Observations (partial):")
        print(f"  agent_0 (Worker1):")
        print(f"    Private: {game.agent_0_obs['private_resources']}")
        print(f"    Public: {game.agent_0_obs['public_resources']}")
        print(f"  agent_1 (Worker2):")
        print(f"    Private: {game.agent_1_obs['private_resources']}")
        print(f"    Public: {game.agent_1_obs['public_resources']}")
        print(f"  agent_2 (Leader):")
        print(f"    Private: {game.agent_2_obs['private_resources']}")
        print(f"    Public: {game.agent_2_obs['public_resources']}")
        
        print(f"\nValue Matrix (score for each task-agent pair):")
        print(f"{'Task':<20}", end="")
        for agent_name in game.agents:
            agent_display_name = game.agents_config[agent_name].get('name', agent_name)
            print(f"{agent_display_name:>15}", end="")
        print()
        print("-" * 65)
        for task_idx, task_name in enumerate(game.tasks):
            print(f"{task_name:<20}", end="")
            for agent_idx, agent_name in enumerate(game.agents):
                score = game.value_matrix[agent_idx, task_idx]
                if game.best_allocation_dict.get(task_name) == agent_name:
                    print(f"{score:>14.2f} *", end="")
                else:
                    print(f"{score:>15.2f}", end="")
            print()
        
        print(f"\nOptimal Allocation:")
        total_score = 0.0
        for task_name, agent_name in game.best_allocation_dict.items():
            agent_idx = game.agents.index(agent_name)
            task_idx = game.tasks.index(task_name)
            score = game.value_matrix[agent_idx, task_idx]
            agent_display_name = game.agents_config[agent_name].get('name', agent_name)
            print(f"  {task_name:<20} -> {agent_display_name:<15} score: {score:.2f}")
            total_score += score
        print(f"  Total score: {total_score:.2f}")
        print(f"  Solver max reward: {game.max_reward:.2f}")
        
        print(f"\nVerification:")
        calculated_reward = game.evaluate_allocation(game.best_allocation_dict)
        print(f"  evaluate_allocation() score: {calculated_reward:.2f}")
        print(f"  Solver max reward: {game.max_reward:.2f}")
        if abs(calculated_reward - game.max_reward) < 1e-6:
            print(f"  PASS: scores match!")
        else:
            print(f"  WARNING: scores mismatch!")
        
        print(f"\n{'='*60}\n")
    
    return games


def final_test_level1(game_folder='task_allocation_demo_01', output_root='benchmark_review', n_level1=None):
    """Batch test Level 1 games: save details and generate statistical report."""
    n_level_1_games = n_level1
    output_root = os.path.join(output_root, game_folder)
    level_1_output_path = f"{output_root}/level1"
    os.makedirs(level_1_output_path, exist_ok=True)
    
    games = load_data_util.load_games(
        target_path=game_folder,
        game_batch_name='level_1_and_2',
        n_games=n_level_1_games
    )
    
    print(f"\n{'='*60}")
    print(f"Final Test - Level 1 Games")
    print(f"{'='*60}")
    print(f"Game folder: {game_folder}")
    print(f"Num games: {len(games)}")
    print(f"Output path: {level_1_output_path}")
    print(f"{'='*60}\n")
    best_allocations = [game.best_allocation_dict for game in games]
    best_values = [game.max_reward for game in games]
    
    
    n_resources = []
    for game in games:
        total_private = sum(
            sum(resources.values()) 
            for resources in game.game_resource_state['agent_private_resources'].values()
        )
        total_public = sum(game.game_resource_state['public_resources'].values())
        n_resources.append(total_private + total_public)
    
    n_tasks = [len(game.tasks) for game in games]
    n_completed_tasks = [len(allocation) for allocation in best_allocations]
    resource_utilizations = []
    for game in games:
        if game.best_allocation is not None and np.sum(game.best_allocation) > 0:
            usage = game.constraint_matrix @ game.best_allocation
            cap = game.vec_resource_state
            mask = cap > 1e-6
            if np.any(mask):
                util = float(np.mean((usage[mask] / cap[mask]).clip(0.0, 1.0)))
                resource_utilizations.append(util)
            else:
                resource_utilizations.append(0.0)
        else:
            resource_utilizations.append(0.0)
    
    print("Saving game details...")
    for i, game in tqdm(enumerate(games), total=len(games), desc="Processing games"):
        with open(f"{level_1_output_path}/game_{i:04d}.txt", 'w', encoding='utf-8') as f:
            f.write(game.gt_representation())
    
    print(f"\n{'='*60}")
    print(f"Statistics Report")
    print(f"{'='*60}")
    print(f"Mean optimal reward: {np.mean(best_values):.2f} +/- {np.std(best_values):.2f}")
    print(f"Mean num tasks: {np.mean(n_tasks):.2f} +/- {np.std(n_tasks):.2f}")
    print(f"Mean completed tasks: {np.mean(n_completed_tasks):.2f} +/- {np.std(n_completed_tasks):.2f}")
    task_completion_rates = [n_completed_tasks[i] / n_tasks[i] if n_tasks[i] > 0 else 0.0 for i in range(len(games))]
    print(f"Mean task completion rate: {np.mean(task_completion_rates):.2%} +/- {np.std(task_completion_rates):.2%}")
    print(f"Mean total resources: {np.mean(n_resources):.2f} +/- {np.std(n_resources):.2f}")
    print(f"Mean resource utilization: {np.mean(resource_utilizations):.2%} +/- {np.std(resource_utilizations):.2%}")
    allocation_counts = {}
    for allocation in best_allocations:
        allocation_str = json.dumps(allocation, sort_keys=True, ensure_ascii=False)
        allocation_counts[allocation_str] = allocation_counts.get(allocation_str, 0) + 1
    
    print(f"\nAllocation statistics:")
    print(f"  Unique allocations: {len(allocation_counts)}")
    if len(allocation_counts) <= 10:
        print(f"  Allocation details:")
        for allocation_str, count in sorted(allocation_counts.items(), key=lambda x: x[1], reverse=True):
            allocation = json.loads(allocation_str)
            print(f"    {allocation}: {count} times")
    task_in_solution_counts = Counter()
    for allocation in best_allocations:
        for task in allocation.keys():
            task_in_solution_counts[task] += 1
    
    if task_in_solution_counts:
        print(f"\nTask frequency in optimal allocations (bias check):")
        total_games_with_solutions = len([a for a in best_allocations if len(a) > 0])
        
        task_freqs = sorted(task_in_solution_counts.items(), key=lambda x: x[1], reverse=True)
        
        frequencies = [count / total_games_with_solutions for _, count in task_freqs] if total_games_with_solutions > 0 else []
        if frequencies:
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            max_freq = max(frequencies)
            min_freq = min(frequencies)
            
            print(f"  Games with solutions: {total_games_with_solutions}")
            print(f"  Mean frequency: {mean_freq:.2%} +/- {std_freq:.2%}")
            print(f"  Max frequency: {max_freq:.2%}, Min frequency: {min_freq:.2%}")
            print(f"  Frequency range: {max_freq - min_freq:.2%}")
            
            if std_freq > 0.15 or (max_freq - min_freq) > 0.3:
                print(f"  WARNING: potential bias detected in task frequency distribution")
            else:
                print(f"  Distribution is relatively uniform, no obvious bias")
        
        print(f"\n  Task frequency details (descending):")
        for task, count in task_freqs:
            freq = count / total_games_with_solutions if total_games_with_solutions > 0 else 0
            print(f"    {task}: {count} times ({freq:.2%})")
    
    print(f"\n{'='*60}\n")
    
    return games


def analyze_database_coverage(game_folder='task_allocation_demo_01', database_root='database_1225', n_games=None):
    """Analyze dataset coverage of the database (task, persona, domain distributions)."""
    games = load_data_util.load_games(
        target_path=game_folder,
        game_batch_name='level_1_and_2',
        n_games=n_games
    )
    
    if len(games) == 0:
        print("No games found!")
        return
    
    domains = get_all_domains(database_root)
    database_info = {}
    all_tasks = set()
    all_personas = set()
    
    for domain_name in domains:
        db = load_database_domain(database_root, domain_name)
        database_info[domain_name] = {
            'tasks': set(db.tasks),
            'personas': {p.get('persona_id') for p in db.agent_pool if p.get('persona_id')},
            'n_tasks': len(db.tasks),
            'n_personas': len(db.agent_pool)
        }
        all_tasks.update(db.tasks)
        all_personas.update(database_info[domain_name]['personas'])
    
    
    task_counts = Counter()
    persona_counts = Counter()
    domain_counts = Counter()
    task_domain_map = defaultdict(set)  # task -> set of domains it appears in
    persona_domain_map = defaultdict(set)  # persona -> set of domains it appears in
    
    for game in games:
        for task in game.tasks:
            task_counts[task] += 1
            if hasattr(game, 'domain_name') and game.domain_name:
                task_domain_map[task].add(game.domain_name)
        
        if hasattr(game, 'persona_ids') and game.persona_ids:
            for agent_name, persona_id in game.persona_ids.items():
                if persona_id:
                    persona_counts[persona_id] += 1
                    if hasattr(game, 'domain_name') and game.domain_name:
                        persona_domain_map[persona_id].add(game.domain_name)
        
        if hasattr(game, 'domain_name') and game.domain_name:
            domain_counts[game.domain_name] += 1
    
    total_games = len(games)
    
    
    print(f"\n{'='*60}")
    print(f"Database Coverage Analysis")
    print(f"{'='*60}")
    print(f"Dataset: {game_folder}")
    print(f"Total games: {total_games}")
    print(f"Database root: {database_root}")
    print(f"{'='*60}\n")
    
    
    print(f"📊 Domain Distribution:")
    for domain_name in domains:
        count = domain_counts.get(domain_name, 0)
        percentage = (count / total_games * 100) if total_games > 0 else 0
        print(f"  {domain_name}: {count} games ({percentage:.1f}%)")
    print()
    
    
    print(f"📋 Task Coverage (per domain):")
    for domain_name in domains:
        db_tasks = database_info[domain_name]['tasks']
        covered_tasks = {task for task in db_tasks if task_counts[task] > 0}
        coverage_rate = len(covered_tasks) / len(db_tasks) * 100 if len(db_tasks) > 0 else 0
        
        print(f"\n  Domain: {domain_name}")
        print(f"    Total tasks in database: {len(db_tasks)}")
        print(f"    Covered tasks: {len(covered_tasks)} ({coverage_rate:.1f}%)")
        print(f"    Uncovered tasks: {len(db_tasks) - len(covered_tasks)}")
        
        
        task_freqs = [(task, task_counts[task]) for task in db_tasks]
        task_freqs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    Task frequency (top 10):")
        for task, freq in task_freqs[:10]:
            percentage = (freq / total_games * 100) if total_games > 0 else 0
            print(f"      {task}: {freq} times ({percentage:.1f}%)")
        
        if len(task_freqs) > 10:
            print(f"    ... and {len(task_freqs) - 10} more tasks")
    
    print()
    
    
    print(f"👥 Persona Coverage (per domain):")
    for domain_name in domains:
        db_personas = database_info[domain_name]['personas']
        covered_personas = {p for p in db_personas if persona_counts[p] > 0}
        coverage_rate = len(covered_personas) / len(db_personas) * 100 if len(db_personas) > 0 else 0
        
        print(f"\n  Domain: {domain_name}")
        print(f"    Total personas in database: {len(db_personas)}")
        print(f"    Covered personas: {len(covered_personas)} ({coverage_rate:.1f}%)")
        print(f"    Uncovered personas: {len(db_personas) - len(covered_personas)}")
        
        
        persona_freqs = [(p, persona_counts[p]) for p in db_personas]
        persona_freqs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"    Persona frequency (top 10):")
        for persona_id, freq in persona_freqs[:10]:
            percentage = (freq / total_games * 100) if total_games > 0 else 0
            print(f"      {persona_id}: {freq} times ({percentage:.1f}%)")
        
        if len(persona_freqs) > 10:
            print(f"    ... and {len(persona_freqs) - 10} more personas")
    
    print()
    
    
    print(f"🎲 Diversity Metrics:")
    
    
    if task_counts:
        task_probs = np.array(list(task_counts.values())) / sum(task_counts.values())
        task_entropy = -np.sum(task_probs * np.log2(task_probs + 1e-10))
        max_task_entropy = np.log2(len(task_counts))
        task_diversity = task_entropy / max_task_entropy if max_task_entropy > 0 else 0
        print(f"  Task diversity (Shannon entropy): {task_entropy:.3f} / {max_task_entropy:.3f} = {task_diversity:.3f}")
    
    
    if persona_counts:
        persona_probs = np.array(list(persona_counts.values())) / sum(persona_counts.values())
        persona_entropy = -np.sum(persona_probs * np.log2(persona_probs + 1e-10))
        max_persona_entropy = np.log2(len(persona_counts))
        persona_diversity = persona_entropy / max_persona_entropy if max_persona_entropy > 0 else 0
        print(f"  Persona diversity (Shannon entropy): {persona_entropy:.3f} / {max_persona_entropy:.3f} = {persona_diversity:.3f}")
    
    
    if domain_counts:
        domain_probs = np.array(list(domain_counts.values())) / sum(domain_counts.values())
        domain_entropy = -np.sum(domain_probs * np.log2(domain_probs + 1e-10))
        max_domain_entropy = np.log2(len(domain_counts))
        domain_diversity = domain_entropy / max_domain_entropy if max_domain_entropy > 0 else 0
        print(f"  Domain diversity (Shannon entropy): {domain_entropy:.3f} / {max_domain_entropy:.3f} = {domain_diversity:.3f}")
    
    
    print(f"\n📈 Overall Coverage:")
    total_db_tasks = sum(db['n_tasks'] for db in database_info.values())
    total_db_personas = sum(db['n_personas'] for db in database_info.values())
    covered_tasks_all = len({task for task in all_tasks if task_counts[task] > 0})
    covered_personas_all = len({p for p in all_personas if persona_counts[p] > 0})
    
    print(f"  Task coverage: {covered_tasks_all} / {total_db_tasks} ({covered_tasks_all/total_db_tasks*100:.1f}%)")
    print(f"  Persona coverage: {covered_personas_all} / {total_db_personas} ({covered_personas_all/total_db_personas*100:.1f}%)")
    
    print(f"\n{'='*60}\n")
    
    return {
        'task_counts': dict(task_counts),
        'persona_counts': dict(persona_counts),
        'domain_counts': dict(domain_counts),
        'task_domain_map': {k: list(v) for k, v in task_domain_map.items()},
        'persona_domain_map': {k: list(v) for k, v in persona_domain_map.items()},
        'diversity_metrics': {
            'task_entropy': task_entropy if task_counts else 0,
            'persona_entropy': persona_entropy if persona_counts else 0,
            'domain_entropy': domain_entropy if domain_counts else 0,
        }
    }


if __name__ == '__main__':
    default_game_folder = 'task_allocation_demo_01'
    
    parser = argparse.ArgumentParser(description='Run task allocation game tests')
    parser.add_argument('--game_folder', type=str, default=default_game_folder,
                        help='Game folder name (default: task_allocation_demo_01)')
    parser.add_argument('--output_root', type=str, default='benchmark_review',
                        help='Output root directory (default: benchmark_review)')
    parser.add_argument('--n_level1', type=int, default=None,
                        help='Number of Level 1 games (default: None = all)')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed info for each game')
    parser.add_argument('--coverage', action='store_true',
                        help='Analyze dataset coverage of database')
    parser.add_argument('--database_root', type=str, default='database_1225',
                        help='Database root under data/task_allocation/source')
    
    args = parser.parse_args()
    
    if args.coverage:
        analyze_database_coverage(
            game_folder=args.game_folder,
            database_root=args.database_root,
            n_games=args.n_level1
        )
    elif args.detailed:
        detailed_test_level1(
            game_folder=args.game_folder,
            n_games=args.n_level1,
            verbose=True
        )
    else:
        final_test_level1(
            game_folder=args.game_folder,
            output_root=args.output_root,
            n_level1=args.n_level1
        )
