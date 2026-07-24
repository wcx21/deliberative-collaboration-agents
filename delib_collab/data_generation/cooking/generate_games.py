import os
import time

import numpy as np
from delib_collab.data_generation.cooking import planner as planning_algos
from delib_collab.data_generation.cooking import io as load_data_util
from delib_collab.data_generation.cooking import game as create_game_utils
from tqdm import tqdm
from delib_collab.data_generation.cooking.planner import state_generate, state_generate_with_partition, solve_planning_with_state, vectorize_state
from delib_collab.data_generation.cooking.io import load_all_data, get_recipes_from_cookbook, get_values_from_cookbook, save_games
from copy import deepcopy


def random_value_disturb(initial_value, disturb_range=3, min_unit=0.05):
    disturbed_value = deepcopy(initial_value.copy())
    n_units = int(disturb_range // min_unit)
    disturb = np.random.randint(low=-n_units, high=n_units + 1, size=len(initial_value)) * min_unit
    disturb = np.round(disturb, 2)
    disturbed_value = disturbed_value + disturb
    return disturbed_value


def select_initial_state(constant_data, game_states, top_k=5, weights=None):
    recipes, values = constant_data
    priorities = []

    for game_state, game_values in game_states:
        if isinstance(game_state, tuple):
            game_state = game_state[0]
            # this means we extract the ground truth state instead of obs form agents

        average_value = (game_values[0] + game_values[1]) / 2
        vec_state = vectorize_state(game_state)
        n_ingredients = np.sum(vec_state)

        best_menu, best_value = solve_planning_with_state(vec_state, recipes, average_value)
        used_ingredients = planning_algos.get_ingredient_requirement_from_menu(best_menu, recipes)
        n_used_ingredients = np.sum(used_ingredients)
        n_dishes = len(best_menu)
        p_used_ingredients = n_used_ingredients / n_ingredients

        priorities.append(p_used_ingredients)

    # choose indexes with top k the highest priorities
    priorities = np.array(priorities)
    print(priorities, np.sort(-priorities)[:top_k])
    selected_indexes = np.argsort(-priorities)[:top_k]

    # construct initial states with top k highest priorities
    selected_initial_states = [game_states[idx] for idx in selected_indexes]
    return selected_initial_states


def gen_game_level_1_dummy_value_ver(data_root, output_dir, n_seed=100, n_game=5, ingredient_num=10, max_unit_num=3,
                                     obs_split_proportion=0.4, priority_weights=None):
    # w_n_dish stands for weight of n_dish in the priority

    ingredients, dishes, cookbook = load_all_data(data_root)
    recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    values = get_values_from_cookbook(dishes, cookbook)
    # recipes is a 2D array
    # values is a 1D array

    print(f"generating random states... (n_seed = {n_seed}) Please wait...")
    game_ingredients_state_gt = [
        create_game_utils.rand_ingredient_state_gt_generation(ingredients, ingredient_num, max_unit_num, weight=None)
        for _ in tqdm(range(n_seed))
    ]
    game_ingredients_obss = [
        create_game_utils.level1_ingredient_partition(ingredients_state_gt, obs_split_proportion)
        for ingredients_state_gt in tqdm(game_ingredients_state_gt)
    ]

    print(f"generating random values... (n_seed = {n_seed}) Please wait...")
    game_values = [
        (random_value_disturb(values), random_value_disturb(values))
        for _ in range(n_seed)
    ]
    print(f"Selecting games ... (n_seed = {n_seed}) Please wait...")

    game_states = list(zip(game_ingredients_state_gt, game_ingredients_obss, game_values))
    database_data = (ingredients, dishes, cookbook)

    processed_game_states = create_game_utils.batch_extract_subset_from_sampled_game(database_data, game_states)

    selected_game_data = create_game_utils.select_initial_state_level1_v1(database_data, processed_game_states,
                                                                          top_k=n_game, priority_weights=priority_weights)

    save_games(output_dir, selected_game_data, game_batch_name='val_games', overwrite=True)

    return


def gen_games_true(data_root, persona_value_folder, output_dir, n_seed=2000, n_level1_game=60,
                l1_ingredient_num=10, max_unit_num=3, l1_obs_split_proportion=0.4,
                l1_priority_weights=None, l1_min_dishes=2,
                p_possible_dish=0.3, p_nonzero_ingredient=0.4, min_possible_dish=5, **kwargs):

    ingredients, dishes, cookbook = load_all_data(data_root)
    database_data = (ingredients, dishes, cookbook)
    load_data_util.sanity_check(ingredients, dishes, cookbook)

    persona_path = os.path.join(data_root, "personas.json")
    personas = load_data_util.load_personas(persona_path)
    personas = load_data_util.load_persona_values(data_root, personas, persona_value_folder)
    persona_obs_probs = load_data_util.load_persona_obs_probs(data_root)

    recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    ingredient_usage, nl_ingredient_usage = create_game_utils.get_ingredient_usage(ingredients, recipes)
    # values = get_values_from_cookbook(dishes, cookbook)
    # recipes is a 2D array
    # values is a 1D array

    # level 1 and level 2 games

    print(f"Ingredient usage in Cookbook: {sorted(nl_ingredient_usage.items(), key=lambda item: item[1], reverse=True)}")

    print(f"Generating for Level 1 and 2")
    print(f"generating random states... (n_seed = {n_seed}) Please wait...")
    ingredient_weight = np.sqrt(ingredient_usage)
    game_ingredients_states_gt = [
        # create_game_utils.rand_ingredient_state_gt_generation(ingredients, l1_ingredient_num, max_unit_num, weight=None)
        create_game_utils.rand_ingredient_state_gt_generation(ingredients, l1_ingredient_num, max_unit_num, weight=ingredient_weight)
        for _ in tqdm(range(n_seed))
    ]
    game_ingredients_obss = [
        create_game_utils.level1_ingredient_partition(ingredients_state_gt, l1_obs_split_proportion)
        for ingredients_state_gt in tqdm(game_ingredients_states_gt)
    ]

    sampled_persona_pairs = create_game_utils.sample_personas(personas, num_personas=2, num_times=n_seed)
    processed_subsets = create_game_utils.batch_extract_subset_from_ingredient_states(
        database_data, game_ingredients_states_gt, p_possible_dish=p_possible_dish,
        p_nonzero_ingredient=p_nonzero_ingredient, subset_func='ext1_1'
    )

    print(f"Creating games (n_seed = {n_seed}) Please wait...")
    time.sleep(0.01)
    games = []
    for i in tqdm(range(n_seed)):
        subset_data = processed_subsets[i]
        persona_pair = sampled_persona_pairs[i]
        ingredients_state = game_ingredients_states_gt[i]
        game_ingredients_obs = game_ingredients_obss[i]

        sub_ingredients, sub_dishes, sub_cookbook = subset_data
        game = create_game_utils.GeneralMenuDesignGame(
            sub_ingredients, sub_dishes, sub_cookbook, persona_pair, persona_obs_probs, ingredients_state,
            l1_obs=game_ingredients_obs
        )
        games.append(game)

    print(f"Selecting games ... (n_seed = {n_seed}) Please wait...")
    # game_states = list(zip(game_ingredients_state_gt, game_ingredients_obss, game_values))
    # processed_game_states = create_game_utils.batch_extract_subset_from_sampled_game(database_data, game_states)
    # selected_game_data = create_game_utils.select_initial_state_level1_v1(database_data, processed_game_states,
    #                                                                       top_k=n_game, priority_weights=priority_weights)

    selected_game_data_l1_and_l2 = create_game_utils.select_games_with_initial_state_v1(
        database_data, games, top_k=n_level1_game, priority_weights=l1_priority_weights, min_dish=l1_min_dishes,
        min_possible_dish=min_possible_dish
    )
    save_games(output_dir, selected_game_data_l1_and_l2, game_batch_name='level_1_and_2', overwrite=True)

    return


if __name__ == '__main__':
    # output_dir = 'data/dev_0_1'
    # gen_game_random(database_root, n_seed=500)

    database_root = 'full_data'
    output_dir = 'test_games'
    persona_value_folder = 'processed/persona_values'

    data_gen_kwargs = {
        "n_seed": 10000,
        "n_level1_game": 60,
        "l1_ingredient_num": 8,
        "max_unit_num": 4,
        "p_possible_dish": 0.4,
        "p_nonzero_ingredient": 0.4,
        "min_possible_dish": 5,
        # "l1_priority_weights": [0.2, 0.2, -0.1]
        "l1_priority_weights": [0.1, 0.05, -0.01],
        "l1_min_dishes": 3,
    }

    # gen_game_level_1_dummy_value_ver(database_root, output_dir, **data_gen_kwargs)
    gen_games_true(database_root, persona_value_folder, output_dir, **data_gen_kwargs)
