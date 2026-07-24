import random
import numpy as np
from scipy.optimize import linprog
import pulp as lp


def state_generate(ingredients, ingredient_ratio=0.3, max_unit_num=5):
    # ingredients: list of strings
    # ingredient_ratio: ratio ingredients to be non-zero
    # max_unit_num: max of number per ingredient
    # output: dictionary with ingredients as keys and their quantities as values

    n_nonzero_ingredients = int(len(ingredients) * ingredient_ratio)
    selected_ingredients = random.sample(ingredients, n_nonzero_ingredients)

    ingredient_dict = {}
    for ingredient in ingredients:
        if ingredient in selected_ingredients:
            ingredient_dict[ingredient] = random.randint(1, max_unit_num)
        else:
            ingredient_dict[ingredient] = 0

    return ingredient_dict


def state_generate_with_partition(ingredients, ingredient_ratio=0.3, max_unit_num=5, min_split_range=0.4):
    # ingredients: list of strings
    # ingredient_ratio: ratio ingredients to be non-zero
    # max_unit_num: max of number per ingredient
    # output: dictionary with ingredients as keys and their quantities as values

    assert 0 <= min_split_range <= 0.5, "min_split_range must be between 0 and 0.5"
    ingredient_dict = state_generate(ingredients, ingredient_ratio, max_unit_num)
    n_ingredients = sum(ingredient_dict.values())

    min_obs_ingredient_num = np.ceil(n_ingredients * min_split_range)
    obs_ingredient_num_1 = np.random.randint(low=min_obs_ingredient_num, high=n_ingredients - min_obs_ingredient_num)
    obs_ingredient_indexes_1 = np.random.choice(range(n_ingredients), size=obs_ingredient_num_1, replace=False)

    ingredient_dict_obs_1 = {k: 0 for k in ingredient_dict}
    ingredient_dict_obs_2 = {k: 0 for k in ingredient_dict}
    ing_counter = 0
    for ingredient in ingredient_dict:
        for _ in range(ingredient_dict[ingredient]):
            if ing_counter in obs_ingredient_indexes_1:
                ingredient_dict_obs_1[ingredient] += 1
            else:
                ingredient_dict_obs_2[ingredient] += 1
            ing_counter += 1

    return ingredient_dict, ingredient_dict_obs_1, ingredient_dict_obs_2


def vectorize_state(ingredient_dict, ingredient_list=None, safe_convert=True):
    if ingredient_list is None:
        return np.array(list(ingredient_dict.values()))

    vec_ingredients = np.zeros(len(ingredient_list))
    for ingre, count in ingredient_dict.items():
        if safe_convert:
            if ingre in ingredient_list:
                vec_ingredients[ingredient_list.index(ingre)] = count
        else:
            vec_ingredients[ingredient_list.index(ingre)] = count
    return vec_ingredients


def constraint_satisfaction(state, recipes, menu):
    # Check if the menu satisfies the constraints
    # That is, the sum of ingredients used must not exceed the available quantities in the state

    total_ingredients_used = np.sum(recipes[menu, :], axis=0)
    return np.all(total_ingredients_used <= state)


def get_possible_dishes(vec_ingredient_state, recipes):
    # ingredient_state: 1D np.array
    # recipes: 2D np.array
    # output: list of dish indexes

    possible_dishes = []
    for dish_index in range(recipes.shape[0]):
        if (vec_ingredient_state >= recipes[dish_index]).all():
            possible_dishes.append(dish_index)

    return possible_dishes


def get_ingredient_requirement_from_menu(menu, recipes):
    # menu: list of dish indexes
    # recipes: a 2D np.array, where recipes[i][j] represents the quantity of ingredient j needed for dish i
    # output: list of ingredients used in the menu

    ingredient_requirement = np.zeros(recipes.shape[1], dtype=int)
    for dish_index in menu:
        ingredient_requirement += recipes[dish_index, :]

    return ingredient_requirement


def check_menu_reward_manually(ingredient_state, recipes, menu, value1, value2, average='average', ingredients=None, dishes=None):
    ingredient_requirement = get_ingredient_requirement_from_menu(menu, recipes)
    if ingredients is not None:
        readable_requirements = {ingredients[i]: ingredient_requirement[i]
                                 for i, v in enumerate(ingredient_requirement) if ingredient_requirement[i] > 0}
        print(f"Required Ingredients: {readable_requirements}")

    diff = (ingredient_state - ingredient_requirement)
    if (diff < 0).any():
        print("Warning: ingredient insufficient for the designed menu")
        return 0

    if average == 'average':
        critria = (value1 + value2) / 2
    elif average == 'add':
        critria = value1 + value2
    else:
        raise ValueError("average must be 'average' or 'add'")

    reward = 0
    for dish_index in menu:
        reward += critria[dish_index]
    return reward


def solve_planning_with_state_bad(state, recipes, values):
    # state: a 1D np.array, where state[i] represents the quantity of ingredient i in the current state
    # recipes: a 2D np.array, where recipes[i][j] represents the quantity of ingredient j needed for dish i
    # values: a 1D array, where values[i] represents the value of dish i
    # output:
    # optimal_menu: a list of dish indexes to cook. Note that each dish can only be cooked once
    # Must follow the constraint, and maximize the total value of the dishes in the menu
    # BAD: This is not integer programming!!

    n_dishes = len(values)
    n_ingredients = len(state)

    # Objective function: maximize the total value of selected dishes (negative for maximization in linprog)
    c = -values  # We use negative because linprog minimizes

    # Constraints: the total usage of each ingredient must not exceed the available quantities in the state
    A_ub = recipes.T  # Shape (n_dishes, n_ingredients)
    b_ub = state  # Shape (n_ingredients,)

    # Bounds: each dish can either be selected (1) or not selected (0)
    x_bounds = [(0, 1) for _ in range(n_dishes)]

    # Linear programming to maximize the total value of the dishes selected
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

    if result.success:
        # Get the indices of the selected dishes
        print(result.x)
        optimal_menu = np.where(result.x >= 0.5)[0].astype(int).tolist()
        return optimal_menu, -result.fun
    else:
        # If the optimization failed, return an empty menu
        return [], 0


def solve_planning_with_state(state, recipes, values):
    # state: a 1D np.array, where state[i] represents the quantity of ingredient i in the current state
    # recipes: a 2D np.array, where recipes[i][j] represents the quantity of ingredient j needed for dish i
    # values: a 1D array, where values[i] represents the value of dish i
    # output:
    # optimal_menu: a list of dish indexes to cook. Note that each dish can only be cooked once
    # Must follow the constraint, and maximize the total value of the dishes in the menu

    n_dishes = len(values)
    n_ingredients = len(state)

    prob = lp.LpProblem("Menu_Optimization", lp.LpMaximize)

    x = [lp.LpVariable(f"x_{i}", cat=lp.LpBinary) for i in range(n_dishes)]

    prob += lp.lpSum([values[i] * x[i] for i in range(n_dishes)])

    for j in range(n_ingredients):
        prob += lp.lpSum([recipes[i][j] * x[i] for i in range(n_dishes)]) <= state[j]

    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    # prob.solve()
    if lp.LpStatus[prob.status] == 'Optimal':
        optimal_menu = [i for i in range(n_dishes) if lp.value(x[i]) is not None and lp.value(x[i]) >= 0.5]
        # optimal_menu = [i for i in range(n_dishes) if lp.value(x[i]) is not None and lp.value(x[i]) >= 0.5]
        # if a value is 0, it will be None, and then obviously we don't need it
        total_value = lp.value(prob.objective)
        return optimal_menu, total_value
    else:
        return [], 0


def local_test_01():
    # Ingredients and recipes for the test
    ingredients = ["rice", "flour", "egg", "milk", "butter"]
    recipes = np.array([[2, 0, 1, 0, 0],
                        [0, 3, 0, 1, 1],
                        [1, 0, 2, 1, 0]])
    values = np.array([10, 15, 12])  # Values for each recipe

    # Generate a state (ingredient quantities)
    ingredient_dict = state_generate(ingredients, ingredient_ratio=1, max_unit_num=4)
    state = vectorize_state(ingredient_dict)
    print(state)

    # Solve the planning problem
    optimal_menu, value = solve_planning_with_state(state, recipes, values)

    # Output the optimal menu and check basic constraints
    print(f"Optimal menu: {optimal_menu}")

    # Check that at least one dish is selected
    assert len(optimal_menu) > 0, "No dish selected!"

    # Check that the chosen menu satisfies the ingredient constraints
    assert constraint_satisfaction(state, recipes, optimal_menu), "Constraints are violated!"


if __name__ == '__main__':
    local_test_01()
