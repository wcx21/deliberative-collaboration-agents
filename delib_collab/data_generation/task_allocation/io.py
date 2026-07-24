"""
Data loading utilities for the task allocation problem.
"""

import os
import sys
import json
import pickle as pkl
import numpy as np
from delib_collab.paths import TASK_ALLOCATION_GAME_DATA_DIR, TASK_ALLOCATION_SOURCE_DATA_DIR


# ==================== Path Setup ====================

task_allocation_root = os.path.dirname(os.path.abspath(__file__))
task_allocation_data_root = str(TASK_ALLOCATION_SOURCE_DATA_DIR)


def _resolve_source_path(root_path):
    if os.path.exists(root_path):
        return root_path
    candidate = os.path.join(str(TASK_ALLOCATION_SOURCE_DATA_DIR), root_path)
    if os.path.exists(candidate):
        return candidate
    return candidate


def _resolve_game_path(target_path):
    if os.path.exists(target_path):
        return target_path
    return os.path.join(str(TASK_ALLOCATION_GAME_DATA_DIR), target_path)


def _register_pickle_aliases():
    from delib_collab.data_generation.task_allocation import game as task_game

    sys.modules.setdefault("task_allocation_create_game_utils", task_game)
    sys.modules.setdefault("benchmark_dev.task_allocation_create_game_utils", task_game)


# ==================== Data Loading ====================


def load_resources(file_path):
    """Load resource type list from a text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        resources = file.read().splitlines()
    resources = [r.strip() for r in resources if r.strip()]
    return resources


def load_tasks(file_path):
    """Load task name list from a text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        tasks = file.read().splitlines()
    tasks = [t.strip() for t in tasks if t.strip()]
    return tasks


def load_task_requirements(file_path):
    """Load task requirements JSON (per-task, per-agent resource needs)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        task_requirements = json.load(file)
    return task_requirements


def load_agents_config(file_path):
    """Load agent configuration JSON (resources, efficiency, etc.)."""
    with open(file_path, 'r', encoding='utf-8') as file:
        agents_config = json.load(file)
    return agents_config


def load_all_data(root_path):
    """Load all data files and return (resources, tasks, task_requirements, agents_config)."""
    root_path = _resolve_source_path(root_path)
    
    resources_path = os.path.join(root_path, 'resources.txt')
    tasks_path = os.path.join(root_path, 'tasks.txt')
    task_requirements_path = os.path.join(root_path, 'task_requirements.json')
    agents_config_path = os.path.join(root_path, 'agents.json')
    
    resources = load_resources(resources_path)
    tasks = load_tasks(tasks_path)
    task_requirements = load_task_requirements(task_requirements_path)
    agents_config = load_agents_config(agents_config_path)
    
    return resources, tasks, task_requirements, agents_config


# ==================== Helpers ====================


def extract_private_and_public_resources(resources, agents_config, task_requirements):
    """Extract and return (private_resource_list, public_resource_list) from config."""
    private_resource_set = set()
    for agent_config in agents_config.values():
        private_resources = agent_config.get('private_resources', {})
        private_resource_set.update(private_resources.keys())
    private_resource_list = [r for r in resources if r in private_resource_set]
    
    public_resource_set = set()
    for task_req in task_requirements.values():
        if 'public_resources' in task_req:
            public_resource_set.update(task_req['public_resources'].keys())
    public_resource_list = [r for r in resources if r in public_resource_set]
    
    return private_resource_list, public_resource_list


# ==================== Data Conversion ====================


def get_requirement_matrix_from_config(task_requirements, tasks, agents, 
                                       private_resource_list, public_resource_list):
    """Build structured requirement matrices from task_requirements config.

    Returns dict with 'private_requirements' (per-agent MxK_priv) and
    'public_requirements' (MxK_pub).
    """
    M = len(tasks)
    K_priv = len(private_resource_list)
    K_pub = len(public_resource_list)
    N = len(agents)
    
    private_requirements = {}
    for agent_idx, agent_name in enumerate(agents):
        private_requirements[agent_name] = np.zeros((M, K_priv), dtype=np.float32)
        for task_idx, task_name in enumerate(tasks):
            if task_name in task_requirements and agent_name in task_requirements[task_name]:
                agent_req = task_requirements[task_name][agent_name]
                for res_idx, res_name in enumerate(private_resource_list):
                    private_requirements[agent_name][task_idx, res_idx] = agent_req.get(res_name, 0)
    
    public_requirements = np.zeros((M, K_pub), dtype=np.float32)
    for task_idx, task_name in enumerate(tasks):
        if task_name in task_requirements and 'public_resources' in task_requirements[task_name]:
            public_req = task_requirements[task_name]['public_resources']
        else:
            public_req = {}
        for res_idx, res_name in enumerate(public_resource_list):
            public_requirements[task_idx, res_idx] = public_req.get(res_name, 0)
    
    return {
        'private_requirements': private_requirements,
        'public_requirements': public_requirements
    }


def get_value_matrix_from_agents(agents_config, tasks, base_value=None):
    """Build NxM value matrix: value_matrix[i,j] = base_value[j] * efficiency[agent_i][task_j]."""
    N = len(agents_config)  # expected to be 3
    M = len(tasks)
    
    
    if base_value is None:
        base_value = np.ones(M, dtype=np.float32)
    else:
        base_value = np.array(base_value, dtype=np.float32)
    
    value_matrix = np.zeros((N, M), dtype=np.float32)
    
    agents = ['agent_0', 'agent_1', 'agent_2']
    for agent_idx, agent_name in enumerate(agents):
        if agent_name not in agents_config:
            continue
        
        agent_config = agents_config[agent_name]
        efficiency = agent_config.get('efficiency', {})
        
        for task_idx, task_name in enumerate(tasks):
            efficiency_value = efficiency.get(task_name, 0.0)
            value_matrix[agent_idx, task_idx] = base_value[task_idx] * efficiency_value
    
    value_matrix = np.round(value_matrix, 2)
    
    return value_matrix


# ==================== Validation ====================


def sanity_check(resources, tasks, task_requirements, agents_config):
    """Validate consistency across resources, tasks, task_requirements, and agents_config."""
    assert len(resources) == len(set(resources)), "Duplicate resources!"
    assert len(tasks) == len(set(tasks)), "Duplicate tasks!"
    
    for task in tasks:
        assert task in task_requirements.keys(), \
            f"Task '{task}' not found in task_requirements!"
    
    for task in task_requirements.keys():
        assert task in tasks, \
            f"Task '{task}' in task_requirements but not in tasks list!"
    
    required_agents = ['agent_0', 'agent_1', 'agent_2']
    for agent_name in required_agents:
        assert agent_name in agents_config, \
            f"Agent '{agent_name}' not found in agents_config!"
    
    for task_name, task_req in task_requirements.items():
        for agent_name in required_agents:
            assert agent_name in task_req, \
                f"Task '{task_name}' missing config for agent '{agent_name}'!"
        assert 'public_resources' in task_req, \
            f"Task '{task_name}' missing public_resources config!"
    
    for agent_name, agent_config in agents_config.items():
        efficiency = agent_config.get('efficiency', {})
        for task in tasks:
            assert task in efficiency, \
                f"Agent '{agent_name}' efficiency missing task '{task}'!"
    
    print(f"All data checks passed! Good to go! :)")
    return True


# ==================== Game Pickle I/O ====================

def save_games(target_path, game_data_list, game_batch_name='default', overwrite=False):
    target_path = _resolve_game_path(target_path)
    target_game_path = os.path.join(target_path, 'games', game_batch_name)
    if os.path.exists(target_game_path) and not overwrite:
        raise FileExistsError(f'Target game path {target_game_path} already exists. Set overwrite=True to overwrite.')

    os.makedirs(target_game_path, exist_ok=True)
    for index, game_data in enumerate(game_data_list):
        game_name = f'game_{index:04d}.pkl'
        with open(os.path.join(target_game_path, game_name), 'wb') as file:
            pkl.dump(game_data, file)


def load_games(target_path, game_batch_name='default', n_games=5, game_idxes=None):
    target_path = _resolve_game_path(target_path)
    target_game_path = os.path.join(target_path, 'games', game_batch_name)
    if not os.path.exists(target_game_path):
        raise FileNotFoundError(f'Target game path {target_game_path} does not exist.')

    game_data_list = []
    _register_pickle_aliases()
    if game_idxes is not None:
        for game_idx in game_idxes:
            game_file_name = f'game_{game_idx:04d}.pkl'
            game_file_path = os.path.join(target_game_path, game_file_name)
            with open(game_file_path, 'rb') as file:
                game_data_list.append(pkl.load(file))
    else:
        game_files = sorted(os.listdir(target_game_path))
        if n_games is not None:
            game_files = game_files[:n_games]
        for fn in game_files:
            with open(os.path.join(target_game_path, fn), 'rb') as file:
                game_data_list.append(pkl.load(file))

    return game_data_list


# ==================== Test ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing task allocation data loading")
    print("=" * 60)
    
    root_path = 'demo_01'
    resources, tasks, task_requirements, agents_config = load_all_data(root_path)
    
    print(f"\nData loaded successfully!")
    print(f"Resources: {resources}")
    print(f"Tasks: {tasks}")
    print(f"Num tasks: {len(tasks)}")
    print(f"Num agents: {len(agents_config)}")
    
    print(f"\nRunning sanity check...")
    sanity_check(resources, tasks, task_requirements, agents_config)
    
    print(f"\nTesting requirement matrix...")
    private_resource_list, public_resource_list = extract_private_and_public_resources(
        resources, agents_config, task_requirements
    )
    print(f"Private resources: {private_resource_list}")
    print(f"Public resources: {public_resource_list}")
    
    requirement_info = get_requirement_matrix_from_config(
        task_requirements, tasks, ['agent_0', 'agent_1', 'agent_2'],
        private_resource_list, public_resource_list
    )
    
    print(f"Private requirement matrix shapes:")
    for agent_name, req_matrix in requirement_info['private_requirements'].items():
        print(f"  {agent_name}: {req_matrix.shape}")
    print(f"Public requirement matrix shape: {requirement_info['public_requirements'].shape}")
    
    print(f"\nTesting value matrix...")
    value_matrix = get_value_matrix_from_agents(agents_config, tasks)
    print(f"Value matrix shape: {value_matrix.shape}")
    print(f"Value matrix:\n{value_matrix}")
    
    print(f"\nAll tests passed!")
