#!/usr/bin/env python
# -*- coding: UTF-8 -*-



import copy

def agent_call_solver(game, ingredient_state, value):
    return game.agent_call_solver(ingredient_state, value)


def agent_call_dish_calculator(game, ingredient_state, value=None):
    return game.agent_call_dish_calculator(ingredient_state)


def agent_call_state_merge(ingredient_states, values):
    new_ingredient_state = copy.deepcopy(ingredient_states[0])
    for ingre, count in ingredient_states[1].items():
        if ingre in new_ingredient_state:
            new_ingredient_state[ingre] += count
        else:
            new_ingredient_state[ingre] = count

    new_value = copy.deepcopy(values[0])
    for dish_name, value in values[1].items():
        if dish_name in new_value:
            new_value[dish_name] = (new_value[dish_name] + value) / 2
        else:
            new_value[dish_name] = value / 2

    return new_ingredient_state, new_value

