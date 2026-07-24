"""
Demo game generation script for the task allocation problem.

Manually constructs a few game instances and saves them as pickle files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delib_collab.data_generation.task_allocation.game import level1_resource_partition, TaskAllocationGame
from delib_collab.data_generation.task_allocation.io import load_all_data
from delib_collab.data_generation.task_allocation.io import save_games


def print_allocation_scores(game, game_name="Game"):
    """Print allocation score analysis to understand why a solution is optimal."""
    print(f"\n{'='*60}")
    print(f"{game_name} - Allocation Score Analysis")
    print(f"{'='*60}")
    
    print("\nValue Matrix (score for each task-agent pair):")
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
            # mark the optimal allocation
            if game.best_allocation_dict.get(task_name) == agent_name:
                print(f"{score:>14.2f} *", end="")
            else:
                print(f"{score:>15.2f}", end="")
        print()
    
    print(f"\nOptimal Allocation Details:")
    print("-" * 65)
    total_score = 0.0
    for task_name, agent_name in game.best_allocation_dict.items():
        agent_idx = game.agents.index(agent_name)
        task_idx = game.tasks.index(task_name)
        score = game.value_matrix[agent_idx, task_idx]
        agent_display_name = game.agents_config[agent_name].get('name', agent_name)
        print(f"  {task_name:<20} -> {agent_display_name:<15} score: {score:.2f}")
        total_score += score
    
    print("-" * 65)
    print(f"  Total score: {total_score:.2f}")
    print(f"  Solver max reward: {game.max_reward:.2f}")
    
    print(f"\nAlternative Allocation Comparison:")
    print("  (examples only; actual optimum is computed by solver)")
    
    print("\n  Option 1: assign each task to highest-scoring agent (ignoring constraints):")
    max_score_no_constraint = 0.0
    allocation_no_constraint = {}
    for task_idx, task_name in enumerate(game.tasks):
        best_agent_idx = game.value_matrix[:, task_idx].argmax()
        best_agent_name = game.agents[best_agent_idx]
        best_score = game.value_matrix[best_agent_idx, task_idx]
        allocation_no_constraint[task_name] = best_agent_name
        max_score_no_constraint += best_score
        agent_display_name = game.agents_config[best_agent_name].get('name', best_agent_name)
        print(f"    {task_name:<20} -> {agent_display_name:<15} score: {best_score:.2f}")
    print(f"    Total score: {max_score_no_constraint:.2f} (may be infeasible due to constraints)")
    
    print(f"\n  Option 2: optimal allocation (respecting constraints):")
    print(f"    Total score: {game.max_reward:.2f} (feasible and optimal)")
    
    print(f"\nResource Constraints:")
    print(f"  Private resources:")
    for agent_name in game.agents:
        agent_display_name = game.agents_config[agent_name].get('name', agent_name)
        private_resources = game.game_resource_state['agent_private_resources'][agent_name]
        print(f"    {agent_display_name}: {private_resources}")
    print(f"  Public resources: {game.game_resource_state['public_resources']}")
    
    print(f"\n{'='*60}\n")


def create_demo_game_1():
    """Create demo game 1: basic scenario with 3 tasks and 3 agents."""
    print("=" * 60)
    print("Creating Demo Game 1: basic scenario")
    print("=" * 60)
    
    print("\n1. Loading base data...")
    resources, tasks, task_requirements, agents_config = load_all_data('demo_01')
    print(f"   Resources: {resources}")
    print(f"   Tasks: {tasks}")
    print(f"   Num tasks: {len(tasks)}")
    print(f"   Num agents: {len(agents_config)}")
    
    print("\n2. Defining resource state (Ground Truth)...")
    agent_private_resources = {
        'agent_0': {'Time': 10, 'GPU': 2},
        'agent_1': {'Time': 8, 'GPU': 1},
        'agent_2': {'Time': 6, 'GPU': 0}
    }
    public_resources = {'Budget': 300}
    
    print(f"   agent_0 private: {agent_private_resources['agent_0']}")
    print(f"   agent_1 private: {agent_private_resources['agent_1']}")
    print(f"   agent_2 private: {agent_private_resources['agent_2']}")
    print(f"   Public resources: {public_resources}")
    
    print("\n3. Partitioning observations...")
    agent_0_obs, agent_1_obs, agent_2_obs = level1_resource_partition(
        agent_private_resources, public_resources, min_split_range=0.4
    )
    
    print(f"   agent_0 obs:")
    print(f"     Private: {agent_0_obs['private_resources']}")
    print(f"     Public: {agent_0_obs['public_resources']}")
    print(f"   agent_1 obs:")
    print(f"     Private: {agent_1_obs['private_resources']}")
    print(f"     Public: {agent_1_obs['public_resources']}")
    print(f"   agent_2 obs:")
    print(f"     Private: {agent_2_obs['private_resources']}")
    print(f"     Public: {agent_2_obs['public_resources']}")
    
    print("\n4. Creating Game object...")
    game = TaskAllocationGame(
        tasks=tasks,
        resources=resources,
        task_requirements=task_requirements,
        agents_config=agents_config,
        resource_state={
            'agent_private_resources': agent_private_resources,
            'public_resources': public_resources
        },
        partial_observations=(agent_0_obs, agent_1_obs, agent_2_obs)
    )
    
    print(f"   Game object created!")
    print(f"   Optimal allocation: {game.best_allocation_dict}")
    print(f"   Max reward: {game.max_reward}")
    
    print_allocation_scores(game, "Demo Game 1")
    
    return game


def create_demo_game_2():
    """Create demo game 2: resource-abundant scenario."""
    print("\n" + "=" * 60)
    print("Creating Demo Game 2: resource-abundant scenario")
    print("=" * 60)
    
    print("\n1. Loading base data...")
    resources, tasks, task_requirements, agents_config = load_all_data('demo_01')
    
    print("\n2. Defining resource state (Ground Truth)...")
    agent_private_resources = {
        'agent_0': {'Time': 15, 'GPU': 3},
        'agent_1': {'Time': 12, 'GPU': 2},
        'agent_2': {'Time': 10, 'GPU': 1}
    }
    public_resources = {'Budget': 400}
    
    print(f"   agent_0 private: {agent_private_resources['agent_0']}")
    print(f"   agent_1 private: {agent_private_resources['agent_1']}")
    print(f"   agent_2 private: {agent_private_resources['agent_2']}")
    print(f"   Public resources: {public_resources}")
    
    print("\n3. Partitioning observations...")
    agent_0_obs, agent_1_obs, agent_2_obs = level1_resource_partition(
        agent_private_resources, public_resources, min_split_range=0.4
    )
    
    print("\n4. Creating Game object...")
    game = TaskAllocationGame(
        tasks=tasks,
        resources=resources,
        task_requirements=task_requirements,
        agents_config=agents_config,
        resource_state={
            'agent_private_resources': agent_private_resources,
            'public_resources': public_resources
        },
        partial_observations=(agent_0_obs, agent_1_obs, agent_2_obs)
    )
    
    print(f"   Game object created!")
    print(f"   Optimal allocation: {game.best_allocation_dict}")
    print(f"   Max reward: {game.max_reward}")
    
    print_allocation_scores(game, "Demo Game 2")
    
    return game


if __name__ == '__main__':
    print("=" * 60)
    print("Task Allocation - Demo Game Generation")
    print("=" * 60)
    
    try:
        game1 = create_demo_game_1()
        game2 = create_demo_game_2()
        
        print("\n" + "=" * 60)
        print("Saving games...")
        print("=" * 60)
        
        output_dir = 'task_allocation_demo_01'
        game_batch_name = 'level_1_and_2'
        
        save_games(
            target_path=output_dir,
            game_data_list=[game1, game2],
            game_batch_name=game_batch_name,
            overwrite=True
        )
        
        print(f"\nGames saved!")
        print(f"   Path: {output_dir}/games/{game_batch_name}/")
        print(f"   Num games: 2")
        print(f"   Files: game_0000.pkl, game_0001.pkl")
        
        print("\n" + "=" * 60)
        print("Verifying saved games...")
        print("=" * 60)
        
        from delib_collab.data_generation.task_allocation.io import load_games
        loaded_games = load_games(
            target_path=output_dir,
            game_batch_name=game_batch_name,
            n_games=2
        )
        
        print(f"   Loaded games: {len(loaded_games)}")
        for i, game in enumerate(loaded_games):
            print(f"\n   Game {i+1}:")
            print(f"     Optimal allocation: {game.best_allocation_dict}")
            print(f"     Max reward: {game.max_reward}")
        
        print("\n" + "=" * 60)
        print("All tasks completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
