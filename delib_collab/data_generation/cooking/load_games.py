from delib_collab.data_generation.cooking import io as load_data_util




def print_human_readable_recipe(cookbook, dishes):
    for dish_idx, dish in enumerate(dishes):
        ingres = cookbook[dish]['ingredients']
        print(f"{dish} Recipe: {cookbook[dish]['ingredients']}")
    return


def load_games(data_root, n_games=10, start=None, end=None, level=1):
    # ingredients, dishes, cookbook = load_all_data(data_root)
    # recipes = get_recipes_from_cookbook(dishes, ingredients, cookbook)
    # initial_values = get_values_from_cookbook(dishes, cookbook)

    if level in [1, 2]:
        game_batch_name = 'level_1_and_2'
    else:
        raise ValueError(f"Invalid level {level}. Only levels 1 and 2 are supported.")

    if start is not None and end is not None:
        game_idxes = list(range(start, end))
        _games = load_data_util.load_games(data_root, game_batch_name, game_idxes=game_idxes)
    else:
        _games = load_data_util.load_games(data_root, game_batch_name, n_games=n_games)

    return _games


if __name__ == '__main__':
    game_folder = 'test_games'
    # init_test(data_root)
    games = load_games(game_folder, n_games=10, level=1)
    print("Done")
