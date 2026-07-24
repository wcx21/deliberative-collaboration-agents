import os
import sys
import json
import numpy as np
import pickle as pkl
from delib_collab.data_generation.cooking.persona import Persona, create_persona_from_dict
from delib_collab.paths import COOKING_GAME_DATA_DIR, COOKING_SOURCE_DATA_DIR


toy_game_data_root = str(COOKING_GAME_DATA_DIR)


def _resolve_source_path(root_path):
    if os.path.exists(root_path):
        return root_path
    for base_dir in (COOKING_SOURCE_DATA_DIR, COOKING_GAME_DATA_DIR):
        candidate = os.path.join(str(base_dir), root_path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(str(COOKING_SOURCE_DATA_DIR), root_path)


def _resolve_game_path(target_path):
    if os.path.exists(target_path):
        return target_path
    return os.path.join(str(COOKING_GAME_DATA_DIR), target_path)


def _register_pickle_aliases():
    from delib_collab.data_generation.cooking import game as cooking_game
    from delib_collab.data_generation.cooking import persona as cooking_persona

    sys.modules.setdefault("create_game_utils", cooking_game)
    sys.modules.setdefault("benchmarks.create_game_utils", cooking_game)
    sys.modules.setdefault("persona", cooking_persona)
    sys.modules.setdefault("benchmarks.persona", cooking_persona)


########  Load Data Utils ######


def load_ingredients(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        ingredients = file.read().splitlines()
    return ingredients


def load_dishes(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dishes = file.read().splitlines()
    return dishes


def load_personas(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        personas = json.load(file)

    personas = [create_persona_from_dict(p) for p in personas]

    return personas


def load_persona_values(root_path, personas, persona_value_folder):
    root_path = _resolve_source_path(root_path)

    persona_values_path = os.path.join(root_path, persona_value_folder)

    persona_values = {}
    for persona in personas:
        p_name = persona.name
        file_path = os.path.join(persona_values_path, f'{p_name}_values.json')
        with open(file_path, 'r', encoding='utf-8') as file:
            p_value = json.load(file)
            persona.values = p_value

    return personas


def load_persona_obs_probs(root_path):
    root_path = _resolve_source_path(root_path)

    persona_values_path = os.path.join(root_path, 'persona_reveal_probs.json')
    with open(persona_values_path, 'r', encoding='utf-8') as file:
        persona_obs_probs = json.load(file)

    persona_obs_probs = list(persona_obs_probs.values())

    return persona_obs_probs



def load_cookbook(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        cookbook = json.load(file)
    return cookbook


def get_recipes_from_cookbook(dishes, ingredients, cookbook):
    recipes = np.zeros((len(dishes), len(ingredients)), dtype=int)
    for dish_idx, dish in enumerate(dishes):
        ingres = cookbook[dish]['ingredients']
        for ingr, quantity in ingres.items():
            recipes[dish_idx, ingredients.index(ingr)] = quantity
    return recipes


def get_values_from_cookbook(dishes, cookbook):
    values = np.zeros(len(dishes), dtype=int)
    for dish_idx, dish in enumerate(dishes):
        value = cookbook[dish]['value']
        values[dish_idx] = value
    return values


def load_all_data(root_path):
    root_path = _resolve_source_path(root_path)

    ingredients_path = os.path.join(root_path, 'ingredients.txt')
    dishes_path = os.path.join(root_path, 'dishes.txt')
    cookbook_path = os.path.join(root_path, 'cookbook.json')

    ingredients = load_ingredients(ingredients_path)
    dishes = load_dishes(dishes_path)
    cookbook = load_cookbook(cookbook_path)

    return ingredients, dishes, cookbook


########  Other Data Utils ######

def sanity_check(ingredients, dishes, cookbook):
    assert len(ingredients) == len(set(ingredients))
    assert len(dishes) == len(set(dishes))

    for dish in dishes:
        assert dish in cookbook.keys()
    for dish in cookbook.keys():
        assert dish in dishes

    cb_ingredients = set()
    for dish, dish_info in cookbook.items():
        cb_ingredients = cb_ingredients.union(dish_info['ingredients'].keys())

    for ingr in cb_ingredients:
        assert ingr in ingredients
    for ingr in ingredients:
        assert ingr in cb_ingredients

    print(f"All data checks passed. Good to go! :)")
    return True



########  Load Game Utils ######


def save_games(target_path, game_data_list, game_batch_name='default', overwrite=False):
    """
    game_data: a dict with many keys, including all things
    :return:
    """
    target_path = _resolve_game_path(target_path)
    target_game_path = os.path.join(target_path, 'games', game_batch_name)
    if os.path.exists(target_game_path) and not overwrite:
        raise FileExistsError(f'Target game path {target_game_path} already exists. Set overwrite=True to overwrite.')

    os.makedirs(target_game_path, exist_ok=True)
    for index, game_data in enumerate(game_data_list):
        game_name = f'game_{index:04d}.pkl'
        with open(os.path.join(target_game_path, game_name), 'wb') as file:
            pkl.dump(game_data, file)

    return


def load_games(target_path, game_batch_name='default', n_games=5, game_idxes=None):
    target_path = _resolve_game_path(target_path)
    target_game_path = os.path.join(target_path, 'games', game_batch_name)
    if not os.path.exists(target_game_path):
        raise FileNotFoundError(f'Target game path {target_game_path} does not exist.')

    game_data_list = []

    if game_idxes is not None:
        for game_idx in game_idxes:
            game_file_name = f'game_{game_idx:04d}.pkl'
            game_file_path = os.path.join(target_game_path, game_file_name)
            _register_pickle_aliases()
            with open(game_file_path, 'rb') as file:
                game_data = pkl.load(file)
                game_data_list.append(game_data)
    else:
        game_files = os.listdir(target_game_path)
        if n_games is not None:
            game_files = game_files[:n_games]
        for fn in game_files:
            _register_pickle_aliases()
            with open(os.path.join(target_game_path, fn), 'rb') as file:
                game_data = pkl.load(file)
                game_data_list.append(game_data)

    return game_data_list


########


if __name__ == '__main__':
    data = load_all_data('demo_01')
    print(data)
