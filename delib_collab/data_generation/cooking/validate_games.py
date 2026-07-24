from delib_collab.data_generation.cooking.load_games import load_games
from delib_collab.data_generation.cooking import io as load_data_util
from delib_collab.data_generation.cooking import planner as planning_algos


def data_check_and_fix(games, out_game_folder, game_level):
    for i, game in enumerate(games):

        assert len(game.ingredients) == len(game.game_ingredients_state)
        if len(game.dishes) != len(game.cookbook):
            print(f"Fixing game_{i}")
            new_dishes = list(set(game.dishes))

            # dish_idx_map = {dish: i for i, dish in enumerate(new_dishes)}
            game.dishes = new_dishes
            game.recipes = load_data_util.get_recipes_from_cookbook(game.dishes, game.ingredients, game.cookbook)
            game.base_value = game._get_base_values()
            game.agent_0_values, game.agent_1_values = game._get_values()

            game.nl_recipes = game._get_nl_recipes()
            game.nl_agent_0_values = game._get_nl_values(game.agent_0_values)
            game.nl_agent_1_values = game._get_nl_values(game.agent_1_values)

            # upper bound with solver
            game.best_menu, new_max_reward = game._run_solver()
            if new_max_reward != game.max_reward and abs(new_max_reward - game.max_reward) > 1e-3:
                print(f"New max reward: {new_max_reward}, old max reward: {game.max_reward}")
            game.possible_dishes = planning_algos.get_possible_dishes(game.vec_ingredients_state, game.recipes)
            game.max_reward = new_max_reward
        else:
            print(f"Skip game_{i} because dishes and cookbook are already the same length.")

    load_data_util.save_games(out_game_folder, games, game_batch_name=f'level_{game_level}', overwrite=True)



if __name__ == '__main__':

    _game_folder = 'local_test_debug_250520'

    _out_game_folder = 'local_test_debug_250520_fixed'
    _game_level = 1

    _games = load_games(_game_folder, n_games=60, level=_game_level)
    data_check_and_fix(_games, _out_game_folder, '1_and_2')

    _games = load_games(_game_folder, n_games=30, level=3)
    data_check_and_fix(_games, _out_game_folder, '3')

    pass
