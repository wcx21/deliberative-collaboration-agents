#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os, sys
import argparse
import pickle
import traceback

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

import numpy as np

from delib_collab.agents.cooking import tools as tools_pool
from delib_collab.agents.cooking.oracle.agent import BuiltOracleAgent
from delib_collab.prompts.cooking.oracle.level1_without_tools import THINK_PROMPT_LEVEL_1_NO_TOOLS
from delib_collab.prompts.cooking.oracle.level1_with_tools import THINK_PROMPT_LEVEL_1_TOOLS
from delib_collab.prompts.cooking.oracle.level2_without_tools import THINK_PROMPT_LEVEL_2_NO_TOOLS, \
    OBSERVE_PROMPT_LEVEL_2_NO_TOOLS
from delib_collab.prompts.cooking.oracle.level2_with_tools import THINK_PROMPT_LEVEL_2_TOOLS, \
    OBSERVE_PROMPT_LEVEL_2_TOOLS
from delib_collab.agents.cooking.tools import agent_call_solver

from delib_collab.agents.cooking.level1.schemas import ObseverFormat
from delib_collab.common.logging_utils import setup_logger

from delib_collab.data_generation.cooking.load_games import load_games

parser = argparse.ArgumentParser(description='Oracle Agent for cooking')
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
parser.add_argument('--level', default='level_1', help='')


def merge_data(data1, data2):
    merged_data = []

    # Create a dictionary for quick lookup based on Name
    data2_dict = {entry['Name']: entry for entry in data2}

    for entry1 in data1:
        name = entry1['Name']
        # Merge data from both lists based on Name
        if name in data2_dict:
            entry2 = data2_dict[name]
            merged_entry = {**entry1}  # Start with all values from data1
            for key, value in entry2.items():
                if isinstance(value, dict):
                    # Merge dictionaries by combining keys from both data1 and data2
                    for sub_key, sub_value in value.items():
                        if sub_value or not merged_entry[key].get(sub_key):
                            merged_entry[key][sub_key] = sub_value
                elif isinstance(value, list):
                    # Merge lists (prefer non-empty lists from data2)
                    merged_entry[key] = value if value else merged_entry[key]
                elif value:  # Use non-empty values from data2
                    merged_entry[key] = value
            merged_data.append(merged_entry)
        else:
            # If no matching entry in data2, just append the entry from data1
            merged_data.append(entry1)

    return merged_data


def merge_obs(dict1, dict2):
    merged_dict = {}

    for key in dict1.keys() | dict2.keys():
        merged_dict[key] = dict1.get(key, 0) + dict2.get(key, 0)

    return merged_dict


def merge_values(list_1, list_2, dishes):
    merged_values = {}
    for i in range(len(list_1)):
        merged_values[dishes[i]] = (list_1[i] + list_2[i]) / 2

    return merged_values


def single_step_log(name, input, output, output_, input_token_count, output_token_count, logger):
    logger.info('{} is called'.format(name))
    logger.info('{} input is {}'.format(name, input))
    logger.info('{} output is {}'.format(name, output))
    logger.info('{} parsed output is {}'.format(name, output_))
    logger.debug("{} input_tokens: {}".format(name, input_token_count))
    logger.debug("{} output_tokens: {}".format(name, output_token_count))


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
    level = args.level

    print(f"Loading games from {game_folder}, {n_games} games, start: {start}, end: {end}...")
    print(f"No tools: {no_tools}")
    if not no_tools:
        file_name = "_with_tools"
        if level == 'level_1':
            prompt_template_think = THINK_PROMPT_LEVEL_1_TOOLS
            games = load_games(game_folder, n_games=n_games, start=start, end=end, level=1)
        elif level == 'level_2':
            prompt_template_think = THINK_PROMPT_LEVEL_2_TOOLS
            prompt_template_observe = OBSERVE_PROMPT_LEVEL_2_TOOLS
            games = load_games(game_folder, n_games=n_games, start=start, end=end, level=2)
    else:
        file_name = "_no_tools"
        if level == 'level_1':
            prompt_template_think = THINK_PROMPT_LEVEL_1_NO_TOOLS
            games = load_games(game_folder, n_games=n_games, start=start, end=end, level=1)
        elif level == 'level_2':
            prompt_template_think = THINK_PROMPT_LEVEL_2_NO_TOOLS
            prompt_template_observe = OBSERVE_PROMPT_LEVEL_2_NO_TOOLS
            games = load_games(game_folder, n_games=n_games, start=start, end=end, level=2)
    print(file_name)
    print(level)
    game_idxes = list(range(start, end)) if start is not None and end is not None else list(range(n_games))
    
    # Build unified directory structure: result/{exp_name}/level_{level_num}_{with_tools|no_tools}/
    result_path = os.path.join(root_dir, "result")
    folder_name = exp_name
    level_num = level.replace('level_', '')
    category_path = f"level_{level_num}_{'with_tools' if not no_tools else 'no_tools'}"
    category_dir = os.path.join(result_path, folder_name, category_path)
    os.makedirs(category_dir, exist_ok=True)
    
    # Path for oracle.pkl in category directory
    oracle_pkl_path = os.path.join(category_dir, "oracle.pkl")
    
    oracle_result = {}
    for i, game in enumerate(games):
        game_idx = game_idxes[i]
        game_dir = os.path.join(category_dir, f"game_{game_idx}")
        os.makedirs(game_dir, exist_ok=True)
        
        # Save the pickle file
        pickle_file_path = os.path.join(game_dir, "short_result.pkl")

        if os.path.exists(pickle_file_path) and not args.override:  # If the file does not exist, create it
            print(f"game{game_idx}, {pickle_file_path} already exists, skipping.")
            continue

        try:
            built_agent = BuiltOracleAgent(model)
            log_folder = os.path.join(exp_name, category_path)
            logger = setup_logger("oracle" + "_game{}_{}".format(game_idx, level), log_folder=log_folder)
            kwargs = {}
            kwargs["person_1_info"] = merge_data(game.agent_0_value_obs_l2, game.agent_1_value_obs_l2)[0]
            kwargs["person_2_info"] = merge_data(game.agent_0_value_obs_l2, game.agent_1_value_obs_l2)[1]
            kwargs["total_ingredients"] = merge_obs(game.agent_0_obs, game.agent_1_obs)
            kwargs["recipes"] = game.nl_recipes
            kwargs["guest_name"] = game.agent_0_value_obs_l2[0]["Name"]
            kwargs['partner_guest_name'] = game.agent_0_value_obs_l2[1]["Name"]
            kwargs['overall_preference'] = merge_values(game.agent_0_values, game.agent_1_values, game.dishes)
            kwargs['best_menu'], reward = agent_call_solver(game, kwargs['total_ingredients'],
                                                            kwargs['overall_preference'])
            suff_dishes, insuff_dish_info = tools_pool.agent_call_dish_calculator(game, kwargs['total_ingredients'])
            kwargs['available_dishes'] = suff_dishes
            kwargs['unavailable_dishes_info'] = insuff_dish_info

            if level == 'level_2':
                kwargs["tags"] = ["thinking_process", "overall_preference"]
                kwargs["agent_name"] = "observe_agent"
                observe_response, observe_response_, observe_input_tokens, observe_output_tokens, prompt_format = (
                    built_agent.creat_subtask_agent(name="oracle", response_format=ObseverFormat,
                                                    prompt_template=prompt_template_observe, logger=logger, **kwargs))
                print(kwargs["overall_preference"])
                print(kwargs["best_menu"])
                kwargs["overall_preference"] = observe_response_["overall_preference"]

                kwargs['best_menu'], reward = agent_call_solver(game, kwargs['total_ingredients'],
                                                                kwargs['overall_preference'])
                print(kwargs["overall_preference"])
                print(kwargs["best_menu"])
                single_step_log("observe", observe_response, observe_response_, observe_response_, observe_input_tokens,
                                observe_output_tokens, logger)

            kwargs["tags"] = ["reasoning_process", "proposal"]
            kwargs["agent_name"] = "think_agent"
            response, response_, input_tokens, output_tokens, prompt_format = (
                built_agent.creat_subtask_agent(name="oracle", response_format=ObseverFormat,
                                                prompt_template=prompt_template_think, logger=logger, **kwargs))
            single_step_log("oracle", response, response_, response_, input_tokens, output_tokens, logger)
            final_proposal = response_["proposal"]["menu_proposal"]

            agent_score = game.evaluate_menu(final_proposal)
            flexible_agent_score = game.evaluate_menu_loose(final_proposal)
            max_reward = game.max_reward
            scores = (agent_score, flexible_agent_score, game.max_reward,
                      np.round(agent_score / max_reward, 3), np.round(flexible_agent_score / max_reward, 3))

            short_result = {
                'final_proposal': final_proposal,
                'scores': scores,
            }

            with open(pickle_file_path, 'wb') as f:
                pickle.dump(short_result, f)
            oracle_result[f"game_{game_idx}"] = short_result

        except Exception as e:
            traceback.print_exc()   # Print the full traceback to the console
            pass
    
    # Save oracle.pkl in category directory
    with open(oracle_pkl_path, 'wb') as f:
        pickle.dump(oracle_result, f)
