#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Task allocation tool functions for agent solver calls."""

import copy
import numpy as np


def agent_call_solver(game, total_resources, overall_preferences):
    """
    Solve optimal task allocation given inferred global resource state and preferences.

    Args:
        game: TaskAllocationGame object
        total_resources: dict, global resource state
        overall_preferences: dict or np.array, global preferences

    Returns:
        allocation: dict, optimal task allocation
        reward: float, score
    """
    if isinstance(overall_preferences, dict):

        agents = game.agents
        tasks = game.tasks
        value_matrix = np.zeros((len(agents), len(tasks)))
        for agent_idx, agent_name in enumerate(agents):
            if agent_name in overall_preferences:
                for task_idx, task_name in enumerate(tasks):
                    if task_name in overall_preferences[agent_name]:
                        value_matrix[agent_idx, task_idx] = overall_preferences[agent_name][task_name]
        overall_preferences = value_matrix
    
    return game.agent_call_solver(total_resources, overall_preferences, natural_language=True)


def agent_call_task_calculator(game, total_resources):
    """
    Compute which tasks can be assigned to which agents based on resource constraints.

    Args:
        game: TaskAllocationGame object
        total_resources: dict, global resource state

    Returns:
        available_allocations: dict, feasible task-agent assignments
        insufficient_allocation_info: dict, infeasible assignments with missing resources
    """
    return game.agent_call_task_calculator(total_resources, natural_language=True)


def agent_call_state_merge(resource_states, preferences):
    """
    Merge multiple agents' resource states and preferences.

    Resource states are merged by taking the max value; preferences are averaged.

    Args:
        resource_states: list of resource state dicts
        preferences: list of preference dicts

    Returns:
        merged_resource_state: dict
        merged_preferences: dict
    """
    if len(resource_states) == 0:
        return {}, {}
    
    # Merge resource states by taking the max (conservative estimate)
    merged_resource_state = copy.deepcopy(resource_states[0])
    
    for resource_state in resource_states[1:]:
        for res_name, res_value in resource_state.get('public_resources', {}).items():
            if res_name in merged_resource_state.get('public_resources', {}):
                merged_resource_state['public_resources'][res_name] = max(
                    merged_resource_state['public_resources'][res_name],
                    res_value
                )
            else:
                merged_resource_state['public_resources'][res_name] = res_value
        


        for agent_name in ['agent_0', 'agent_1', 'agent_2']:
            if agent_name in resource_state.get('agent_private_resources', {}):
                agent_resources = resource_state['agent_private_resources'][agent_name]
                if agent_name in merged_resource_state.get('agent_private_resources', {}):
                    for res_name, res_value in agent_resources.items():
                        if res_name in merged_resource_state['agent_private_resources'][agent_name]:
                            merged_resource_state['agent_private_resources'][agent_name][res_name] = max(
                                merged_resource_state['agent_private_resources'][agent_name][res_name],
                                res_value
                            )
                        else:
                            merged_resource_state['agent_private_resources'][agent_name][res_name] = res_value
                else:
                    merged_resource_state['agent_private_resources'][agent_name] = copy.deepcopy(agent_resources)
    
    # Merge preferences by averaging
    merged_preferences = copy.deepcopy(preferences[0])
    
    for preference in preferences[1:]:
        for agent_name in ['agent_0', 'agent_1', 'agent_2']:
            if agent_name in preference:
                if agent_name in merged_preferences:
                    for task_name, task_value in preference[agent_name].items():
                        if task_name in merged_preferences[agent_name]:
                            merged_preferences[agent_name][task_name] = (
                                merged_preferences[agent_name][task_name] + task_value
                            ) / 2
                        else:
                            merged_preferences[agent_name][task_name] = task_value / len(preferences)
                else:
                    merged_preferences[agent_name] = {
                        task_name: task_value / len(preferences)
                        for task_name, task_value in preference[agent_name].items()
                    }
    
    return merged_resource_state, merged_preferences

