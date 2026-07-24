import os

from tqdm import tqdm
import numpy as np

from delib_collab.data_generation.cooking import io as load_data_util
from delib_collab.data_generation.cooking import load_games as load_games_v2
from delib_collab.data_generation.cooking import planner as planning_algos
from delib_collab.data_generation.cooking import legacy_load_games as load_games
from delib_collab.data_generation.cooking.planner import state_generate, solve_planning_with_state, vectorize_state
from delib_collab.data_generation.cooking.io import load_all_data, get_recipes_from_cookbook, get_values_from_cookbook


def init_test(data_root, n_repeat=1):
    ingredients, dishes, cookbook = load_all_data(data_root)
    recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    values = get_values_from_cookbook(dishes, cookbook)

    for _ in range(n_repeat):
        print(f"repeat count {_+1}:\n")
        game_ingredients_state = state_generate(ingredients, ingredient_ratio=0.3, max_unit_num=3)
        game_vec_state = vectorize_state(game_ingredients_state)

        menu = solve_planning_with_state(game_vec_state, recipes, values)

        print(f"Ingredients: {game_ingredients_state}")
        print(f"Solved Menu: {[dishes[d_idx] for d_idx in menu]}")
    return


def formal_test(data_root, game_batch_name='dev_v0', n_games=5):
    ingredients, dishes, cookbook = load_all_data(data_root)
    recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    initial_values = get_values_from_cookbook(dishes, cookbook)

    game_states = load_data_util.load_games(data_root, game_batch_name, n_games=n_games)
    print(f"Formal Test on {game_batch_name} with {n_games} games:\n")
    print(f"ALL Ingredients: {ingredients}")
    print(f"ALL Dishes: {dishes}")
    load_games.print_human_readable_recipe(cookbook, dishes)

    for i, game_state in enumerate(game_states):
        game_ingredients, game_values = game_state
        game_ingredients_state, obs_0, obs_1 = game_ingredients
        print(f"\nGame {i}:")
        nonzero_ingredients = {k: v for k, v in game_ingredients_state.items() if v > 0}
        nonzero_ingredient_obs_0 = {k: v for k, v in obs_0.items() if v > 0}
        nonzero_ingredient_obs_1 = {k: v for k, v in obs_1.items() if v > 0}

        human_readable_value = [{dishes[i]: v for i, v in enumerate(game_value)} for game_value in game_values]
        print(f"Agent 1 Values: {human_readable_value[0]}")
        print(f"Agent 2 Values: {human_readable_value[1]}")
        print(f"Agent 1 obs: {nonzero_ingredient_obs_0}")
        print(f"Agent 2 obs: {nonzero_ingredient_obs_1}")
        print(f"Ingredient State: {nonzero_ingredients}")

        game_vec_state = vectorize_state(game_ingredients_state)
        average_value = (game_values[0] + game_values[1]) / 2
        best_menu, best_value = solve_planning_with_state(game_vec_state, recipes, average_value)
        calculated_reward = planning_algos.check_menu_reward_manually(
            game_vec_state, recipes, best_menu, game_values[0], game_values[1], ingredients=ingredients, dishes=dishes,
        )
        print(f"Solved Menu: {[dishes[d_idx] for d_idx in best_menu]}")
        print(f"solver value: {best_value}, menu reward: {calculated_reward}\n")
    return


def formal_test_v2(data_root, game_batch_name='val_games', n_games=5):

    games = load_data_util.load_games(data_root, game_batch_name, n_games=n_games)
    print(f"Formal Test on {game_batch_name} with {n_games} games:\n")
    # load_games.print_human_readable_recipe(cookbook, dishes)

    for i, game in enumerate(games):
        game_data, game_state = game
        ingredients, dishes, cookbook = game_data
        recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)

        game_ingredients_state_gt, game_obss, game_values = game_state

        obs_0, obs_1 = game_obss
        print(f"\nGame {i}:")
        nonzero_ingredients = {k: v for k, v in game_ingredients_state_gt.items() if v > 0}
        nonzero_ingredient_obs_0 = {k: v for k, v in obs_0.items() if v > 0}
        nonzero_ingredient_obs_1 = {k: v for k, v in obs_1.items() if v > 0}

        human_readable_value = [{dishes[i]: v for i, v in enumerate(game_value)} for game_value in game_values]
        print(f"Agent 1 Values: {human_readable_value[0]}")
        print(f"Agent 2 Values: {human_readable_value[1]}")
        print(f"Agent 1 obs: {nonzero_ingredient_obs_0}")
        print(f"Agent 2 obs: {nonzero_ingredient_obs_1}")
        print(f"Ingredient State: {nonzero_ingredients}")

        game_vec_state = vectorize_state(game_ingredients_state_gt)
        average_value = (game_values[0] + game_values[1]) / 2
        best_menu, best_value = solve_planning_with_state(game_vec_state, recipes, average_value)
        calculated_reward = planning_algos.check_menu_reward_manually(
            game_vec_state, recipes, best_menu, game_values[0], game_values[1], ingredients=ingredients, dishes=dishes,
        )
        print(f"Solved Menu: {[dishes[d_idx] for d_idx in best_menu]}")
        print(f"solver value: {best_value}, menu reward: {calculated_reward}\n")
    return


def final_test_0213(game_folder='test_games', output_root='benchmark_review'):
    # level 1 and 2 test
    n_level_1_games = 60
    output_root = os.path.join(output_root, game_folder)
    level_12_output_path = f"{output_root}/level12"
    os.makedirs(level_12_output_path, exist_ok=True)

    games = load_games_v2.load_games(game_folder, n_games=n_level_1_games, level=1)
    print(f"Final Test on {game_folder} with {n_level_1_games} level 1 and 2 games :\n")
    best_menus = [game.best_menu for game in games]
    best_values = [game.max_reward for game in games]
    for i, game in tqdm(enumerate(games)):
        best_menu, best_value = game.best_menu, game.max_reward
        print(f"Game {i}: Solved Menu: {[game.dishes[dish_idx] for dish_idx in best_menu]}, Best Value: {best_value}")

        with open(f"{level_12_output_path}/game_{i}.txt", 'w') as f:
            f.write(game.gt_representation())

    print(f"Average Best Value: {np.mean(best_values)} +- {np.std(best_values)}")
    menu_sizes = [len(menu) for menu in best_menus]
    print(f"Average Menu Size: {np.mean(menu_sizes)} +- {np.std(menu_sizes)}")


if __name__ == '__main__':
    # data_root = 'data/dev_0_1'
    # game_folder = 'dev_0_1'
    # game_batch_name = 'val_games'
    # # init_test(data_root)
    # formal_test_v2(game_folder, game_batch_name, n_games=10)

    game_folder = 'test_games'
    final_test_0213(game_folder)



