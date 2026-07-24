#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Task allocation Level 1 entry script."""

import os
import sys
import argparse
from dotenv import load_dotenv

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.data_generation.task_allocation.io import load_games

from delib_collab.agents.task_allocation.level1.process_with_tools import ProcessTaskAllocationLevel1WithToolV1
from delib_collab.agents.task_allocation.level1 import process_without_tools

from delib_collab.agents.task_allocation.level1.metrics import (
    aggregate_metrics, save_metrics_summary, print_aggregated_metrics
)

parser = argparse.ArgumentParser(description='Process Task Allocation Level 1')
parser.add_argument('-g', '--game_folder', type=str, default='task_allocation_demo_01', 
                    help='game folder (default: task_allocation_demo_01)')
parser.add_argument('-n', '--n_games', type=int, default=1, help='number of games')
parser.add_argument('-s', '--start', type=int, default=None, help='start of games')
parser.add_argument('-e', '--end', type=int, default=None, help='end of games')
parser.add_argument('--max_round', type=int, default=6, help='maximum conversation rounds')
parser.add_argument('--max_character', type=int, default=3000, help='maximum characters per round')
parser.add_argument('-m', '--model', type=str, default="model-name", help='LLM model name')
parser.add_argument('--exp_name', type=str, default='debug_task_allocation', help='experiment name')
parser.add_argument('-o', '--override', default=False, action='store_true', 
                    help='override existing results')
parser.add_argument('--no_tools', default=False, action='store_true', 
                    help='do not use solver tools')


def load_task_allocation_games(game_folder, n_games=None, start=None, end=None):
    """Load task allocation game objects."""
    
    if start is not None and end is not None:
        game_idxes = list(range(start, end))
        games = load_games(
            target_path=game_folder,
            game_batch_name='level_1_and_2',
            n_games=None,
            game_idxes=game_idxes
        )
    else:
        game_idxes = list(range(n_games)) if n_games else None
        games = load_games(
            target_path=game_folder,
            game_batch_name='level_1_and_2',
            n_games=n_games,
            game_idxes=None
        )
        if n_games:
            games = games[:n_games]
            game_idxes = list(range(len(games)))
    
    return games, game_idxes


if __name__ == '__main__':
    args = parser.parse_args()
    game_folder = args.game_folder
    n_games = args.n_games
    start = args.start
    end = args.end
    max_round = args.max_round
    max_character = args.max_character
    model = args.model
    exp_name = args.exp_name
    override = args.override
    no_tools = args.no_tools

    print(f"="*60)
    print(f"Task Allocation Level 1 Entry")
    print(f"="*60)
    print(f"Loading games from {game_folder}, start: {start}, end: {end}...")
    print(f"Model: {model}, exp_name: {exp_name}, override: {override}, no_tools: {no_tools}")
    print(f"Max Round: {max_round}, Max Character: {max_character}")
    
    try:
        games, game_idxes = load_task_allocation_games(
            game_folder=game_folder,
            n_games=n_games,
            start=start,
            end=end
        )
        print(f"Successfully loaded {len(games)} games.")
    except Exception as e:
        print(f"Error loading games: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    all_game_metrics = []
    
    for i, game in enumerate(games):
        game_idx = game_idxes[i] if game_idxes else i
        
        print(f"\n{'='*60}")
        print(f"Processing Game {game_idx}")
        print(f"{'='*60}")
        
        print(f"  Tasks: {game.tasks}")
        print(f"  Agents: {game.agents}")
        print(f"  Max Reward: {game.max_reward}")
        print(f"  Best Allocation: {game.best_allocation_dict}")
        
        
        if not no_tools:
            process_level_1 = ProcessTaskAllocationLevel1WithToolV1(
                game, 
                LLM_model_name=model, 
                max_conversation_rounds=max_round, 
                max_character=max_character,
                log_folder=os.path.join(exp_name, 'task_allocation_level_1_with_tools'), 
                log_name='v1', 
                game_id=game_idx,
                record_folder_name=os.path.join(exp_name, 'task_allocation_level_1_with_tools', f'game_{game_idx}'), 
                game_level='level_1'
            )
        else:
            process_level_1 = process_without_tools.ProcessTaskAllocationLevel1V1(
                game, 
                LLM_model_name=model, 
                max_conversation_rounds=max_round, 
                max_character=max_character,
                log_folder=os.path.join(exp_name, 'task_allocation_level_1_no_tools'), 
                log_name='v1', 
                game_id=game_idx,
                record_folder_name=os.path.join(exp_name, 'task_allocation_level_1_no_tools', f'game_{game_idx}'), 
                game_level='level_1'
            )
        
        if override or not os.path.exists(process_level_1.short_result_file_path):
            print(f"\nStarting negotiation for Game {game_idx}...")
            try:
                final_proposal, score, game_metrics = process_level_1.process
                print(f"\nGame {game_idx} completed.")
                print(f"  Final Proposal: {final_proposal}")
                print(f"  Score: {score}")
                
                all_game_metrics.append(game_metrics)
            except Exception as e:
                print(f"\n❌ Error processing Game {game_idx}: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Skipping Game {game_idx} and continuing with next game...")
                default_metrics = {
                    'final_reward': 0.0,
                    'max_reward': game.max_reward if hasattr(game, 'max_reward') else 0.0,
                    'nr': 0.0,
                    'nar': 0.0,
                    'is_valid': False,
                    'original_allocation_size': 0,
                    'submenu_size': 0,
                    'submenu_reward': 0.0,
                    'error': str(e)
                }
                all_game_metrics.append(default_metrics)
                continue
        else:
            print(f"Skip game {game_idx}, already processed.")
            print(f"  Result file: {process_level_1.short_result_file_path}")
            
            try:
                import pickle
                with open(process_level_1.short_result_file_path, 'rb') as f:
                    short_result = pickle.load(f)
                    if 'evaluation_metrics' in short_result:
                        all_game_metrics.append(short_result['evaluation_metrics'])
                    elif 'game_metrics' in short_result:
                        all_game_metrics.append(short_result['game_metrics'])
                    else:
                        print(f"  Warning: No evaluation_metrics or game_metrics found in existing result, skipping metrics collection.")
            except Exception as e:
                print(f"  Warning: Failed to load existing metrics: {e}")
    
    print(f"\n{'='*60}")
    print(f"All games processed.")
    print(f"{'='*60}")
    
    if all_game_metrics:
        print(f"\nCalculating aggregated metrics...")
        
        aggregated_metrics = aggregate_metrics(all_game_metrics)
        print_aggregated_metrics(aggregated_metrics)

        if not no_tools:
            metrics_output_path = os.path.join('logs', exp_name, 'task_allocation_level_1_with_tools', 'metrics_summary.json')
        else:
            metrics_output_path = os.path.join('logs', exp_name, 'task_allocation_level_1_no_tools', 'metrics_summary.json')
        
        try:
            save_metrics_summary(all_game_metrics, aggregated_metrics, metrics_output_path)
            print(f"\nMetrics summary saved to:")
            print(f"  JSON: {metrics_output_path}")
            print(f"  TXT:  {metrics_output_path.replace('.json', '.txt')}")
            if os.path.exists(metrics_output_path):
                print(f"  ✅ JSON file confirmed at: {os.path.abspath(metrics_output_path)}")
            else:
                print(f"  ⚠️  WARNING: JSON file not found at: {os.path.abspath(metrics_output_path)}")
        except Exception as e:
            print(f"\n❌ Error saving metrics summary: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nNo metrics collected (all games were skipped or failed).")
    
    print(f"\n{'='*60}")
    print(f"Experiment completed.")
    print(f"{'='*60}")
