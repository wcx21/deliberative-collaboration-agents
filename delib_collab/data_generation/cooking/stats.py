import numpy as np
from delib_collab.data_generation.cooking import planner as planning_algos
from delib_collab.data_generation.cooking import io as load_data_util
from tqdm import tqdm
from delib_collab.data_generation.cooking.planner import state_generate, state_generate_with_partition, solve_planning_with_state, vectorize_state
from delib_collab.data_generation.cooking.io import load_all_data, get_recipes_from_cookbook, get_values_from_cookbook, save_games
from copy import deepcopy


def stat_ingredient_usage(ingredients, dishes, cookbook, recipes, values):
    # check how many dishes each ingredient is used in
    ingredient_usage = np.zeros(len(ingredients), dtype=np.int32)
    print(np.sum(recipes))
    for i in range(len(ingredients)):
        usage = recipes[:, i]
        n_usage = np.sum(usage > 0)
        ingredient_usage[i] = n_usage
        print(ingredients[i], n_usage)

    nl_ingredient_usage = {ingredients[i]: ingredient_usage[i] for i in range(len(ingredients))}
    print(nl_ingredient_usage)

    print("Number of dishes using each ingredient:")
    for ingredient, usage in nl_ingredient_usage.items():
        print(f"{ingredient}: {usage}")

    return nl_ingredient_usage


def main(data_root):
    ingredients, dishes, cookbook = load_all_data(data_root)
    recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    values = get_values_from_cookbook(dishes, cookbook)
    print(f"n_dishes = {len(dishes)}, n_ingredients = {len(ingredients)}")

    # recipes is a 2D array
    # values is a 1D array

    print("stat ingredient usage...")
    nl_ingredient_usage = stat_ingredient_usage(ingredients, dishes, cookbook, recipes, values)

    return


if __name__ == '__main__':
    data_root = 'database/dev_0_eng'
    main(data_root)
