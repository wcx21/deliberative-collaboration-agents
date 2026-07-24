#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os, sys
import argparse

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.data_generation.cooking.load_games import load_games
from delib_collab.agents.cooking.level1.process_with_tools import ProcessLevel1WithToolV1
from delib_collab.agents.cooking.level1 import process_without_tools

parser = argparse.ArgumentParser(description='Process Level 1')
parser.add_argument('-g', '--game_folder', type=str, default='test_games', help='game folder')
parser.add_argument('-n', '--n_games', type=int, default=5, help='number of games')
parser.add_argument('-s', '--start', type=int, default=None, help='start of games')
parser.add_argument('-e', '--end', type=int, default=None, help='end of games')
parser.add_argument('--max_round', type=int, default=6, help='')
parser.add_argument('--max_character', type=int, default=3000, help='')
parser.add_argument('-m', '--model', type=str, default="model-name", help='LLM')
parser.add_argument('--exp_name', type=str, default='tmp', help='')
parser.add_argument('-o', '--override', default=False, action='store_true', help='')
parser.add_argument('--no_tools', default=False, action='store_true', help='')


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

    print(f"Loading games from {game_folder}, start: {start}, end: {end}...")
    print(f"Model: {model}, exp_name: {exp_name}, override: {override}, no_tools: {no_tools}")
    print(f"Max Round: {max_round}, Max Character: {max_character}")
    games = load_games(game_folder, n_games=n_games, start=start, end=end, level=1)
    game_idxes = list(range(start, end)) if start is not None and end is not None else list(range(n_games))

    for i, game in enumerate(games):
        game_idx = game_idxes[i]
        if not no_tools:
            process_level_1_v0 = ProcessLevel1WithToolV1(
                game, LLM_model_name=model, max_conversation_rounds=max_round, max_character=max_character,
                log_folder=os.path.join(exp_name, 'level_1_with_tools'), log_name='v1', game_id=game_idx,
                record_folder_name=os.path.join(exp_name, 'level_1_with_tools', f'game_{game_idx}'), game_level='level_1'
            )
        else:
            process_level_1_v0 = process_without_tools.ProcessLevel1V1(
                game, LLM_model_name=model, max_conversation_rounds=max_round, max_character=max_character,
                log_folder=os.path.join(exp_name, 'level_1_no_tools'), log_name='v1', game_id=game_idx,
                record_folder_name=os.path.join(exp_name, 'level_1_no_tools', f'game_{game_idx}'), game_level='level_1'
            )
        if override or not os.path.exists(process_level_1_v0.short_result_file_path):
            final_proposal, score = process_level_1_v0.process
        else:
            print(f"Skip game {game_idx}, already processed.")
