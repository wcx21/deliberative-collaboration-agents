from delib_collab.data_generation.cooking import io as load_data_util
from delib_collab.data_generation.cooking import planner as planning_algos
from delib_collab.data_generation.cooking.planner import state_generate, solve_planning_with_state, vectorize_state
from delib_collab.data_generation.cooking.io import load_all_data, get_recipes_from_cookbook, get_values_from_cookbook


class LevelOneGame:
    '''
    dishes: a list of dish names
    ingredients: a list of ingredient names
    cookbook: a nested dictionary as cookbook
    recipes: a 2D numpy array of recipes, shape (n_dishes, n_ingredients)
    game_ingredients_state: a dict of ingredient quantities, better used with vectorized_state
    agent_0_values: a 1D numpy array of values for agent 0, shape (n_dishes)
    agent_1_values: a 1D numpy array of values for agent 1, shape (n_dishes)
    '''

    def __init__(self, ingredients, dishes, cookbook, game_state):
        self.ingredients = ingredients
        self.dishes = dishes
        self.cookbook = cookbook
        self.recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)

        game_ingredients, (agent_0_obs, agent_1_obs), (agent_0_values, agent_1_values) = game_state
        # game_ingredients_state, agent_0_obs, agent_1_obs = game_ingredients
        self.game_ingredients_state = game_ingredients
        self.vec_ingredients_state = vectorize_state(self.game_ingredients_state)
        self.agent_0_values = agent_0_values
        self.agent_1_values = agent_1_values
        self.agent_0_obs = agent_0_obs
        self.agent_1_obs = agent_1_obs

        self.nl_recipes = self._get_nl_recipes()
        self.nl_agent_0_values = self._get_nl_values(self.agent_0_values)
        self.nl_agent_1_values = self._get_nl_values(self.agent_1_values)

        # upper bound with solver
        self.best_menu, self.max_reward = self._run_solver()
        self.possible_dishes = planning_algos.get_possible_dishes(self.vec_ingredients_state, self.recipes)

    def _get_nl_recipes(self):
        recipes = dict()
        for dish_idx, dish in enumerate(self.dishes):
            ingres = self.cookbook[dish]['ingredients']
            recipes[dish] = ingres.copy()  # To avoid modifying the original cookbook
        return recipes

    def _get_nl_values(self, values):
        nl_values = dict()
        for dish_idx, dish in enumerate(self.dishes):
            nl_values[dish] = values[dish_idx]
        return nl_values

    def _run_solver(self):
        gt_value = (self.agent_0_values + self.agent_1_values) / 2.0
        menu, reward = planning_algos.solve_planning_with_state(self.vec_ingredients_state, self.recipes, gt_value)
        assert menu is not None and len(menu) > 0, "Error: No solution found in gt solver"
        return menu, reward

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
                if planning_algos.constraint_satisfaction(self.vec_ingredients_state, self.recipes,
                                                          current_menu + [dish_idx]):
                    current_menu.append(dish_idx)
            reward = planning_algos.check_menu_reward_manually(self.vec_ingredients_state, self.recipes, current_menu,
                                                               self.agent_0_values, self.agent_1_values)

        return reward


def print_human_readable_recipe(cookbook, dishes):
    for dish_idx, dish in enumerate(dishes):
        ingres = cookbook[dish]['ingredients']
        print(f"{dish} Recipe: {cookbook[dish]['ingredients']}")
    return


# def load_game_level_1_old(data_root, game_batch_name='dev_v0', n_games=5):
#     ingredients, dishes, cookbook = load_all_data(data_root)
#     recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
#     initial_values = get_values_from_cookbook(dishes, cookbook)
#
#     game_states = load_data_util.load_games(data_root, game_batch_name, n_games=n_games)
#
#     games = [LevelOneGame(ingredients, dishes, cookbook, game_state) for game_state in game_states]
#     return games


def load_game_level_1(data_root, game_batch_name='val_games', n_games=10):
    # ingredients, dishes, cookbook = load_all_data(data_root)
    # recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    # initial_values = get_values_from_cookbook(dishes, cookbook)

    games = load_data_util.load_games(data_root, game_batch_name, n_games=n_games)
    game_objects = []
    for i, game in enumerate(games):
        game_data, game_state = game
        ingredients, dishes, cookbook = game_data
        # recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
        # game_ingredients_state_gt, game_obss, game_values = game_state

        game_objects.append(
            LevelOneGame(ingredients, dishes, cookbook, game_state)
        )
    return game_objects


if __name__ == '__main__':
    game_folder = 'dev_0_1'
    game_batch_name = 'val_games'
    # init_test(data_root)
    load_game_level_1(game_folder, game_batch_name, n_games=10)
