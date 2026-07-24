import numpy as np
import random
import itertools
import json

from tqdm import tqdm
from delib_collab.data_generation.cooking import planner as planning_algos
from delib_collab.data_generation.cooking import io as load_data_util
from delib_collab.data_generation.cooking.planner import  solve_planning_with_state, vectorize_state
from copy import deepcopy

### ================dummy generation for dev 01=====================


def random_value_disturb(initial_value, disturb_range=3, min_unit=0.05):
    disturbed_value = deepcopy(initial_value.copy())
    n_units = int(disturb_range // min_unit)
    disturb = np.random.randint(low=-n_units, high=n_units + 1, size=len(initial_value)) * min_unit
    disturb = np.round(disturb, 2)
    disturbed_value = disturbed_value + disturb
    return disturbed_value


def select_initial_state_v0(constant_data, game_states, top_k=5, weights=None):
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


### ================utils=====================


def extract_subset_by_ingredients_ext1(selected_ingredients, ingredients, dishes, cookbook, recipes,
                                       p_nz_ingredients=0.7, p_ps_dish=0.4):

    # p_nz_ingredients = proportion of non-zero ingredients in the game state
    # p_ps_dish = proportion of possible dishes in the game state

    related_dishes = set()
    for ingredient in selected_ingredients:
        related_dishes = related_dishes.union({dish for dish in dishes if ingredient in cookbook[dish]['ingredients']})

    ext1_ingredients = set()
    for dish in related_dishes:
        for ingredient in cookbook[dish]['ingredients']:
            if ingredient not in selected_ingredients:
                ext1_ingredients.add(ingredient)

    new_ingredients = list(ext1_ingredients) + selected_ingredients
    new_cookbook = {dish: info for dish, info in cookbook.items() if dish in related_dishes}
    new_dishes = list(related_dishes)
    return new_ingredients, new_dishes, new_cookbook


def extract_subset_by_ingredients_ext1_1(game_state_gt, ingredients, dishes, cookbook, recipes,
                                         p_possible_dish=0.3, p_nonzero_ingredient=0.4, min_dishes=10):

    # p_nonzero_ingredient can not be used, because must load all relative ingredients

    game_state_gt = {k: v for k, v in game_state_gt.items() if v > 0}

    vec_state = planning_algos.vectorize_state(game_state_gt, ingredient_list=ingredients)
    possible_dishes = planning_algos.get_possible_dishes(vec_state, recipes)
    possible_dish_nl = [dishes[idx] for idx in possible_dishes]

    n_possible_dishes = len(possible_dishes)
    n_total_dishes = int(max(np.ceil(n_possible_dishes / p_possible_dish), min_dishes))
    n_need_dishes = n_total_dishes - n_possible_dishes

    # n_nonzero_ingredients = len(game_state_gt)
    # n_total_ingredients = int(np.ceil(n_nonzero_ingredients / p_nonzero_ingredient))
    # n_need_ingredients = n_total_ingredients - n_nonzero_ingredients

    nonzero_ingredients = list(game_state_gt.keys())
    related_dishes = set()
    for ingredient in nonzero_ingredients:
        related_dishes = related_dishes.union({dish for dish in dishes if ingredient in cookbook[dish]['ingredients']})

    # related_dishes = related_dishes.difference(set(possible_dish_nl))
    related_dishes = [dish for dish in related_dishes if dish not in possible_dish_nl]

    if n_need_dishes > len(related_dishes):
        extend_dishes = related_dishes
    else:
        # extend_dishes = random.sample(dishes, n_need_dishes)
        extend_dishes = random.sample(related_dishes, n_need_dishes)

    new_dishes = list(set(possible_dish_nl + extend_dishes))

    ext1_ingredients = set()
    for dish in new_dishes:
        for ingredient in cookbook[dish]['ingredients']:
            if ingredient not in game_state_gt:
                ext1_ingredients.add(ingredient)

    # if n_need_ingredients > len(ext1_ingredients):
    #     extend_ingredients = ext1_ingredients
    # else:
    #     extend_ingredients = random.sample(ingredients, n_need_ingredients)
    extend_ingredients = ext1_ingredients

    new_ingredients = list(extend_ingredients) + list(game_state_gt.keys())
    new_cookbook = {dish: info for dish, info in cookbook.items() if dish in new_dishes}

    return new_ingredients, new_dishes, new_cookbook


### ================True Generation Funcs=====================

def rand_ingredient_state_gt_generation(ingredients, ingredient_num=10, max_unit_num=3, weight=None):
    # ingredients: list of strings
    # ingredient_ratio: ratio ingredients to be non-zero
    # max_unit_num: max of number per ingredient
    # output: dictionary with ingredients as keys and their quantities as values

    n_nonzero_ingredients = int(ingredient_num)
    if weight is not None:
        p = np.array(weight) / np.sum(weight)
    else:
        p = None
    selected_ingredients = np.random.choice(ingredients, n_nonzero_ingredients, replace=False, p=p)

    ingredient_dict = {}
    for ingredient in ingredients:
        if ingredient in selected_ingredients:
            ingredient_dict[ingredient] = random.randint(1, max_unit_num)

    return ingredient_dict


def level1_ingredient_partition(ingredients_state, min_split_range=0.4):
    # ingredients: list of strings
    # ingredient_ratio: ratio ingredients to be non-zero
    # max_unit_num: max of number per ingredient
    # output: dictionary with ingredients as keys and their quantities as values

    assert 0 <= min_split_range <= 0.5, "min_split_range must be between 0 and 0.5"
    ingredient_dict = ingredients_state
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

    return ingredient_dict_obs_1, ingredient_dict_obs_2


def batch_extract_subset_from_sampled_game(database, game_states, subset_func='ext1', p_nz_ingredients=0.7,
                                           p_ps_dish=0.4):
    # p_nz_ingredients = proportion of non-zero ingredients in the game state
    # p_ps_dish = proportion of possible dishes in the game state

    ingredients, dishes, cookbook = database
    recipes = load_data_util.get_recipes_from_cookbook(dishes, ingredients, cookbook)

    processed_games = []
    for game_state_gt, game_obss, game_values in game_states:
        selected_ingredients = list(game_state_gt.keys())
        if subset_func == 'ext1':
            subset_data = extract_subset_by_ingredients_ext1(selected_ingredients, ingredients, dishes, cookbook, recipes)
        else:
            raise NotImplementedError(f"subset_func {subset_func} is not implemented")
        sub_ingredients, sub_dishes, sub_cookbook = subset_data

        subset_game_state_gt = {k: game_state_gt[k] if k in game_state_gt else 0 for k in sub_ingredients}
        subset_game_obs_0 = {k: game_obss[0][k] if k in game_state_gt else 0 for k in sub_ingredients}
        subset_game_obs_1 = {k: game_obss[1][k] if k in game_state_gt else 0 for k in sub_ingredients}
        sub_game_values_0 = np.array([game_values[0][dishes.index(dish)] for dish in sub_dishes])
        sub_game_values_1 = np.array([game_values[1][dishes.index(dish)] for dish in sub_dishes])

        new_game_data = subset_data
        new_game_state = subset_game_state_gt, (subset_game_obs_0, subset_game_obs_1), (sub_game_values_0, sub_game_values_1)
        processed_games.append((new_game_data, new_game_state))

    return processed_games


def batch_extract_subset_from_ingredient_states(database, game_state_gts, subset_func='ext1', p_possible_dish=0.3,
                                                p_nonzero_ingredient=0.4):
    # p_nz_ingredients = proportion of non-zero ingredients in the game state
    # p_ps_dish = proportion of possible dishes in the game state

    ingredients, dishes, cookbook = database
    recipes = load_data_util.get_recipes_from_cookbook(dishes, ingredients, cookbook)

    processed_subsets = []
    for game_state_gt in game_state_gts:
        selected_ingredients = list(game_state_gt.keys())
        if subset_func == 'ext1':
            subset_data = extract_subset_by_ingredients_ext1(selected_ingredients, ingredients, dishes, cookbook, recipes)
        elif subset_func == 'ext1_1':
            subset_data = extract_subset_by_ingredients_ext1_1(
                game_state_gt, ingredients, dishes, cookbook, recipes, p_possible_dish, p_nonzero_ingredient
            )

        else:
            raise NotImplementedError(f"subset_func {subset_func} is not implemented")
        processed_subsets.append(subset_data)

    return processed_subsets


def select_initial_state_level1_v1(database, games, top_k=5, range_p_dishes=None, range_p_ingredients=None,
                                   priority_weights=None):
    priorities = []
    priority_details = []

    for game_data, game_state in tqdm(games):
        sub_ingredients, sub_dishes, sub_cookbook = game_data
        sub_recipes = load_data_util.get_recipes_from_cookbook(sub_dishes, sub_ingredients, sub_cookbook)
        game_state_gt, game_obss, game_values = game_state
        # we use ground truth state instead of obs form agents

        nonzero_ingredients = [ingre for ingre, cnt in game_state_gt.items() if cnt > 0]
        vec_state = vectorize_state(game_state_gt)
        n_ingredients = np.sum(vec_state)
        average_value = (game_values[0] + game_values[1]) / 2

        possible_dishes = planning_algos.get_possible_dishes(vec_state, sub_recipes)
        p_nonzero_ingredients = len(nonzero_ingredients) / len(game_state_gt)
        p_possible_dishes = len(possible_dishes) / len(sub_dishes)

        best_menu, best_value = solve_planning_with_state(vec_state, sub_recipes, average_value)
        used_ingredients = planning_algos.get_ingredient_requirement_from_menu(best_menu, sub_recipes)
        n_used_ingredients = np.sum(used_ingredients)
        p_used_ingredients = n_used_ingredients / n_ingredients
        n_dishes = len(best_menu)

        if priority_weights is None:
            priority_weights = [0.5, 0.5, 1]
        w_n_total_ingre, w_n_total_dish, w_n_final_dish = priority_weights
        if p_possible_dishes < 0.1:
            priority = -100
            priority_details.append((p_used_ingredients, p_nonzero_ingredients, p_possible_dishes, 0))
        else:
            if n_dishes > len(possible_dishes):
                print(n_dishes, possible_dishes)
            priority = p_used_ingredients + w_n_total_ingre * p_nonzero_ingredients
            priority += w_n_total_dish * p_possible_dishes + w_n_final_dish * n_dishes / len(possible_dishes)
            priority_details.append((p_used_ingredients, p_nonzero_ingredients, p_possible_dishes, n_dishes / len(possible_dishes)))
        priorities.append(priority)

    # choose indexes with top k the highest priorities
    priorities = np.array(priorities)
    selected_indexes = np.argsort(-priorities)[:top_k]
    print(selected_indexes, -np.sort(-priorities)[:top_k])
    print(f"priority details: {[priority_details[i] for i in selected_indexes]}")

    # construct initial states with top k highest priorities
    selected_initial_states = [games[idx] for idx in selected_indexes]
    return selected_initial_states


def select_games_with_initial_state_v1(database, games, top_k=5, min_possible_dish=5,
                                   min_dish=3, priority_weights=None, reward_weight=0.01):
    priorities = []
    priority_details = []
    priority_details2 = []

    # game is an object of GeneralMenuDesignGame
    for game in tqdm(games):
        sub_ingredients, sub_dishes, sub_cookbook = game.ingredients, game.dishes, game.cookbook
        sub_recipes = load_data_util.get_recipes_from_cookbook(sub_dishes, sub_ingredients, sub_cookbook)
        game_state_gt = game.game_ingredients_state
        game_values_0 = game.agent_0_values
        game_values_1 = game.agent_1_values

        # we use ground truth state instead of obs form agents

        nonzero_ingredients = [ingre for ingre, cnt in game_state_gt.items() if cnt > 0]
        vec_state = vectorize_state(game_state_gt, ingredient_list=sub_ingredients)
        n_ingredients = np.sum(vec_state)
        average_value = (game_values_0 + game_values_1) / 2

        possible_dishes = planning_algos.get_possible_dishes(vec_state, sub_recipes)
        p_nonzero_ingredients = len(nonzero_ingredients) / len(game_state_gt)
        p_possible_dishes = len(possible_dishes) / len(sub_dishes)

        best_menu, best_value = game.best_menu, game.max_reward
        used_ingredients = planning_algos.get_ingredient_requirement_from_menu(best_menu, sub_recipes)
        n_used_ingredients = np.sum(used_ingredients)
        p_used_ingredients = n_used_ingredients / n_ingredients
        n_dishes = len(best_menu)

        if priority_weights is None:
            priority_weights = [0.1, 0.05, 0.03]
        w_n_total_ingre, w_n_possible, w_n_final_dish = priority_weights

        if p_possible_dishes < 0.1:
            priority = -100
            priority_details.append((p_used_ingredients, p_nonzero_ingredients, p_possible_dishes, 0))
            priority_details2.append((p_used_ingredients, p_nonzero_ingredients, p_possible_dishes, 0))
        else:
            if n_dishes > len(possible_dishes):
                print(n_dishes, possible_dishes)
            priority = p_used_ingredients + w_n_total_ingre * p_nonzero_ingredients
            priority += w_n_possible * len(possible_dishes) + w_n_final_dish * n_dishes
            priority_details.append((p_used_ingredients, p_nonzero_ingredients, len(possible_dishes), n_dishes))
            priority_details2.append((p_used_ingredients, w_n_total_ingre * p_nonzero_ingredients,
                                      w_n_possible * p_possible_dishes, w_n_final_dish * n_dishes, reward_weight * best_value))

            priority += reward_weight * best_value

        if n_dishes < min_dish or len(possible_dishes) < min_possible_dish:
            priority = -100

        priorities.append(priority)

    # choose indexes with top k the highest priorities
    priorities = np.array(priorities)
    selected_indexes = np.argsort(-priorities)[:top_k]
    print(selected_indexes, -np.sort(-priorities)[:top_k])
    print(f"p_used_ingredients, p_nonzero_ingredients, len(possible_dishes), n_dishes")
    print(f"priority details: {[priority_details[i] for i in selected_indexes]}")
    print(f"priority details2: {[priority_details2[i] for i in selected_indexes]}")

    # construct initial states with top k highest priorities
    selected_initial_states = [games[idx] for idx in selected_indexes]
    return selected_initial_states


def get_ingredient_usage(ingredients, recipes):
    # check how many dishes each ingredient is used in
    ingredient_usage = np.zeros(len(ingredients), dtype=np.float32)
    for i in range(len(ingredients)):
        usage = recipes[:, i]
        n_usage = np.sum(usage > 0)
        ingredient_usage[i] = n_usage

    nl_ingredient_usage = {ingredients[i]: ingredient_usage[i] for i in range(len(ingredients))}

    return ingredient_usage, nl_ingredient_usage


def sample_personas(personas, num_personas=2, num_times=1):
    persona_perturbs = list(itertools.permutations(personas, num_personas))
    sample_pairs = random.choices(persona_perturbs, k=num_times)
    # Note that personas should be read-only objects, so we do not use deepcopy here.

    return sample_pairs


class GeneralMenuDesignGame:
    COMPLEXITY_BONUS_COEF = 0.2

    '''
    functions start with 'agent' are tools available for LLM agents
    '''

    def __init__(self, ingredients, dishes, cookbook, persona_pair, persona_obs_probs, ingredient_state, l1_obs=None):
        # hard information
        self.ingredients = ingredients
        self.dishes = dishes
        self.cookbook = cookbook
        self.recipes = load_data_util.get_recipes_from_cookbook(dishes, ingredients, cookbook)

        self.personas = persona_pair
        self.persona_0, self.persona_1 = persona_pair
        self.persona_obs_probs = persona_obs_probs
        # extend ingredient
        game_ingredients_state = {
            ingre: ingredient_state[ingre] if ingre in ingredient_state else 0 for ingre in ingredients
        }
        self.game_ingredients_state = game_ingredients_state

        self.vec_ingredients_state = vectorize_state(self.game_ingredients_state, self.ingredients)

        # derived information
        if l1_obs is not None:
            agent_0_obs, agent_1_obs = l1_obs
        else:
            agent_0_obs, agent_1_obs = level1_ingredient_partition(ingredient_state)

        self.agent_0_obs = agent_0_obs
        self.agent_1_obs = agent_1_obs

        self.base_value = self._get_base_values()
        self.agent_0_values, self.agent_1_values = self._get_values()

        self.partial_persona_0_0, self.partial_persona_0_1, self.partial_persona_1_0, self.partial_persona_1_1 = self._get_level2_partial_persona()
        self.agent_0_value_obs_l2 = [self.partial_persona_0_0.get_full_profile(), self.partial_persona_1_0.get_full_profile()]
        self.agent_1_value_obs_l2 = [self.partial_persona_0_1.get_full_profile(), self.partial_persona_1_1.get_full_profile()]

        self.nl_recipes = self._get_nl_recipes()
        self.nl_agent_0_values = self._get_nl_values(self.agent_0_values)
        self.nl_agent_1_values = self._get_nl_values(self.agent_1_values)

        # upper bound with solver
        self.best_menu, self.max_reward = self._run_solver()
        self.possible_dishes = planning_algos.get_possible_dishes(self.vec_ingredients_state, self.recipes)

    def _get_base_values(self):
        # base score is correlated with complexity and ingredient count

        base_value = self.recipes.sum(axis=-1).astype(np.float32)
        base_value += self.COMPLEXITY_BONUS_COEF * base_value ** 1.5
        base_value = np.round(base_value, 2).astype(np.float32)  # Round to nearest integer for simplicity
        return base_value

    def _get_values(self):
        agent_0_values, agent_1_values = [], []
        for i, dish in enumerate(self.dishes):
            agent_0_values.append(self.persona_0.values[dish]['rating'] * self.base_value[i])
            agent_1_values.append(self.persona_1.values[dish]['rating'] * self.base_value[i])

        agent_0_values, agent_1_values = np.array(agent_0_values), np.array(agent_1_values)
        agent_0_values, agent_1_values = np.round(agent_0_values, 2), np.round(agent_1_values, 2)
        return agent_0_values, agent_1_values

    def _get_level2_partial_persona(self):
        # base score is correlated with complexity and ingredient count
        partial_persona_0_0 = self.persona_0.partial_observation(self.persona_obs_probs[0])
        partial_persona_0_1 = self.persona_0.partial_observation(self.persona_obs_probs[1])
        partial_persona_1_0 = self.persona_1.partial_observation(self.persona_obs_probs[0])
        partial_persona_1_1 = self.persona_1.partial_observation(self.persona_obs_probs[1])

        return partial_persona_0_0, partial_persona_0_1, partial_persona_1_0, partial_persona_1_1

    def _get_nl_recipes(self):
        recipes = dict()
        # for dish_idx, dish in enumerate(self.dishes):
        #     ingres = self.cookbook[dish]['ingredients']
        #     recipes[dish] = ingres.copy()  # To avoid modifying the original cookbook

        for dish_idx, dish in enumerate(self.dishes):
            ingres = {self.ingredients[i]: v for i, v in enumerate(self.recipes[dish_idx]) if v > 0}
            recipes[dish] = ingres.copy()  # To avoid modifying the original cookboo

        return recipes


    def _get_nl_values(self, values):
        nl_values = dict()
        for dish_idx, dish in enumerate(self.dishes):
            nl_values[dish] = np.round(values[dish_idx], 2)
        return nl_values

    def _run_solver(self, test_mode=False):
        gt_value = (self.agent_0_values + self.agent_1_values) / 2.0
        menu, reward = planning_algos.solve_planning_with_state(self.vec_ingredients_state, self.recipes, gt_value)
        if menu is None or len(menu) == 0:
            if test_mode:
                raise ValueError("Error: No solution found in gt solver")
        return menu, reward

    def agent_call_solver(self, ingredient_state, value, natural_language=True):
        # functional tool, could be called by agents and should not reveal confidential information
        # ingredient_state and value are in dict form
        # This function solve the best menu given the observation
        # return natural language menu and a reward

        vec_state = vectorize_state(ingredient_state, self.ingredients)
        vec_value = np.zeros(len(self.dishes))
        for i, dish in enumerate(self.dishes):
            vec_value[i] = value.get(dish, 0)

        menu, reward = planning_algos.solve_planning_with_state(vec_state, self.recipes, vec_value)
        if natural_language:
            menu = [self.dishes[dish_idx] for dish_idx in menu]
        # it is possible that no solution is found
        return menu, reward

    def agent_call_dish_calculator(self, ingredient_state, natural_language=True):
        # functional tool, could be called by agents and should not reveal confidential information
        # ingredient_state and value are in dict form
        # This function suggest available dishes, and calculate how much ingredients are needed if not available
        # return available dishes and currently not available ones
        # not available ones: a dict, keys are dish names, values are ingredient needed
        # template: dish_name: need <num> more <ingredient>,

        vec_state = vectorize_state(ingredient_state, self.ingredients)
        available_dish = planning_algos.get_possible_dishes(vec_state, recipes=self.recipes)
        available_dish = [self.dishes[dish_idx] for dish_idx in available_dish]

        insufficient_dish_idxes = [i for i, dish in enumerate(self.dishes) if dish not in available_dish]
        req_info = []
        for idx in insufficient_dish_idxes:
            dish_name = self.dishes[idx]
            req = self.recipes[idx] - vec_state
            req_str = f"{dish_name}: need "
            req_str += ', '.join([f"{int(num)} more {self.ingredients[i]}" for i, num in enumerate(req) if num > 0]) + '.'
            req_info.append(req_str)

        nl_requirement = '\n'.join(req_info) if req_info else ''

        return available_dish, nl_requirement


    def evaluate_menu(self, menu):
        menu = menu.copy()  # To avoid modifying the original menu
        if isinstance(menu, list) and len(menu) > 0 and isinstance(menu[0], str):
            indexed_menu = []
            for dish in menu:
                if dish in self.dishes:
                    indexed_menu.append(self.dishes.index(dish))
                else:
                    print(f"Error: Invalid dish '{dish}' in the menu")
        else:
            indexed_menu = menu

        for dish_idx in indexed_menu:
            if dish_idx not in self.possible_dishes:
                print(f"Warning: Impossible dish {self.dishes[dish_idx]} '{dish_idx}' in the menu")

        reward = planning_algos.check_menu_reward_manually(self.vec_ingredients_state, self.recipes, indexed_menu,
                                                  self.agent_0_values, self.agent_1_values)
        return reward

    def evaluate_menu_loose(self, menu):
        menu = menu.copy()  # To avoid modifying the original menu
        if isinstance(menu, list) and len(menu) > 0 and isinstance(menu[0], str):
            indexed_menu = []
            for dish in menu:
                if dish in self.dishes:
                    indexed_menu.append(self.dishes.index(dish))
                else:
                    print(f"Error: Invalid dish '{dish}' in the menu")
        else:
            indexed_menu = menu

        indexed_menu = [dish_idx for dish_idx in indexed_menu if dish_idx in self.possible_dishes]
        if planning_algos.constraint_satisfaction(self.vec_ingredients_state, self.recipes, indexed_menu):
            reward = planning_algos.check_menu_reward_manually(self.vec_ingredients_state, self.recipes, indexed_menu,
                                                               self.agent_0_values, self.agent_1_values)
        else:
            current_menu = []
            for dish_idx in indexed_menu:
                if planning_algos.constraint_satisfaction(self.vec_ingredients_state, self.recipes, current_menu + [dish_idx]):
                    current_menu.append(dish_idx)
            reward = planning_algos.check_menu_reward_manually(self.vec_ingredients_state, self.recipes, current_menu,
                                                               self.agent_0_values, self.agent_1_values)

        return reward

    def get_feasible_sub_menu(self, menu, verbose=True):
        menu = menu.copy()  # To avoid modifying the original menu
        if isinstance(menu, list) and len(menu) > 0 and isinstance(menu[0], str):
            indexed_menu = []
            for dish in menu:
                if dish in self.dishes:
                    indexed_menu.append(self.dishes.index(dish))
                elif verbose:
                    print(f"Error: Invalid dish '{dish}' in the menu")
        else:
            indexed_menu = menu

        indexed_menu = [dish_idx for dish_idx in indexed_menu if dish_idx in self.possible_dishes]
        if planning_algos.constraint_satisfaction(self.vec_ingredients_state, self.recipes, indexed_menu):
            sub_menu = menu
        else:
            current_menu = []
            for dish_idx in indexed_menu:
                if planning_algos.constraint_satisfaction(self.vec_ingredients_state, self.recipes, current_menu + [dish_idx]):
                    current_menu.append(dish_idx)
            sub_menu = [self.dishes[dish_idx] for dish_idx in current_menu]

        return sub_menu

    def gt_representation(self):
        repr_str = ''
        # repr_str += 'Persona 0: \n'
        # repr_str += json.dumps(self.persona_0.get_full_profile(), indent=4) + '\n'
        # repr_str += 'Persona 1: \n'
        # repr_str += json.dumps(self.persona_1.get_full_profile(), indent=4) + '\n'
        repr_str += 'Ingredient State: \n'
        repr_str += json.dumps(self.game_ingredients_state, indent=4) + '\n'
        repr_str += 'Recipe: \n'
        repr_str += json.dumps(self.nl_recipes, indent=4) + '\n'
        repr_str += 'Values: \n'
        repr_str += json.dumps(self._get_nl_values((self.agent_0_values + self.agent_1_values) / 2), indent=4) + '\n'
        repr_str += 'Possible Dishes: \n'
        repr_str += json.dumps([self.dishes[dish_idx] for dish_idx in self.possible_dishes], indent=4) + '\n'
        repr_str += 'Best Menu: \n'
        repr_str += json.dumps([self.dishes[dish_idx] for dish_idx in self.best_menu], indent=4) + '\n'
        repr_str += 'Max Reward: \n'
        repr_str += str(self.max_reward) + '\n'

        return repr_str

