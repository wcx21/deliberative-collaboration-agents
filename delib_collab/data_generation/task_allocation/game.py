"""
Game creation utilities for the task allocation problem.

Defines TaskAllocationGame and related helpers for game instantiation.
"""

import numpy as np
import json
from copy import deepcopy

from delib_collab.data_generation.task_allocation import planner as task_allocation_planning_algos
from delib_collab.data_generation.task_allocation import io as task_allocation_load_data_util
from delib_collab.data_generation.task_allocation.planner import (
    solve_task_allocation_with_state,
    vectorize_task_allocation_state,
    build_constraint_matrix,
    build_value_vector,
    check_allocation_constraints,
    evaluate_allocation,
    allocation_vector_to_dict,
    dict_to_allocation_vector
)


### ================ Partial Observation Partitioning =====================


def level1_resource_partition(agent_private_resources, public_resources, 
                              min_split_range=0.4, leader_public_ratio=1.0):
    """
    Partition resources into partial observations for Level 1.

    Each agent sees only their own private resources. Public resources are split
    among agents such that their observations sum to the true total.

    Returns:
        (agent_0_obs, agent_1_obs, agent_2_obs)
    """
    assert 0 <= min_split_range <= 0.5, "min_split_range must be between 0 and 0.5"
    assert 0.0 <= leader_public_ratio <= 1.0, "leader_public_ratio must be between 0 and 1"

   
    agent_0_obs = {
        'private_resources': {},
        'public_resources': {}
    }
    agent_1_obs = {
        'private_resources': {},
        'public_resources': {}
    }
    agent_2_obs = {
        'private_resources': {},
        'public_resources': {}
    }

    agent_0_obs['private_resources'] = deepcopy(agent_private_resources.get('agent_0', {}))
    agent_1_obs['private_resources'] = deepcopy(agent_private_resources.get('agent_1', {}))
    agent_2_obs['private_resources'] = deepcopy(agent_private_resources.get('agent_2', {}))
    for res_name, res_value in public_resources.items():
        if res_value == 0:
            agent_0_obs['public_resources'][res_name] = 0
            agent_1_obs['public_resources'][res_name] = 0
            agent_2_obs['public_resources'][res_name] = 0
            continue

        leader_num = int(res_value * leader_public_ratio)
        remain = res_value - leader_num

        if remain > 0:
            min_obs_num = max(1, int(np.ceil(remain * min_split_range)))
            if remain > 2 * min_obs_num:
                obs_num_0 = np.random.randint(
                    low=min_obs_num,
                    high=remain - min_obs_num + 1
                )
            else:
                obs_num_0 = remain // 2
            obs_num_1 = remain - obs_num_0
        else:
            obs_num_0 = 0
            obs_num_1 = 0

        agent_0_obs['public_resources'][res_name] = obs_num_0
        agent_1_obs['public_resources'][res_name] = obs_num_1
        agent_2_obs['public_resources'][res_name] = leader_num

    return agent_0_obs, agent_1_obs, agent_2_obs


### ================ Game Class =====================


class TaskAllocationGame:
    """Encapsulates a task allocation game instance with partial observability."""
    
    def __init__(self, tasks, resources, task_requirements, agents_config,
                 resource_state, partial_observations=None, task_values=None):
        """Initialize a task allocation game instance."""
        self.tasks = tasks
        self.resources = resources
        self.agents = ['agent_0', 'agent_1', 'agent_2']
        self.persona_ids = None
        self.task_requirements = task_requirements
        self.agents_config = agents_config
        self.task_values = task_values  # dict {task_name: value} or None
        
        self.private_resource_list, self.public_resource_list = \
            task_allocation_load_data_util.extract_private_and_public_resources(
                resources, agents_config, task_requirements
            )
        
        self.game_resource_state = resource_state
        agent_private_resources = resource_state['agent_private_resources']
        public_resources = resource_state['public_resources']
        
        self.vec_resource_state = vectorize_task_allocation_state(
            agent_private_resources, public_resources,
            self.private_resource_list, self.public_resource_list
        )
        
        if partial_observations is not None:
            agent_0_obs, agent_1_obs, agent_2_obs = partial_observations
        else:
            agent_0_obs, agent_1_obs, agent_2_obs = level1_resource_partition(
                agent_private_resources, public_resources
            )
        
        self.agent_0_obs = agent_0_obs
        self.agent_1_obs = agent_1_obs
        self.agent_2_obs = agent_2_obs
        
        self.value_matrix = self._build_value_matrix()
        self.constraint_matrix = self._build_constraint_matrix()
        self.best_allocation, self.max_reward = self._run_solver()
        self.best_allocation_dict = allocation_vector_to_dict(
            self.best_allocation, self.tasks, self.agents
        )
    
    def _build_value_matrix(self):
        """Build the N x M value matrix from agent efficiencies and task base values."""
        
        if self.task_values is None:
            base_value = np.ones(len(self.tasks), dtype=np.float32)
        else:
            base_value = np.array([float(self.task_values.get(t, 1.0)) for t in self.tasks], dtype=np.float32)
        
        value_matrix = task_allocation_load_data_util.get_value_matrix_from_agents(
            self.agents_config, self.tasks, base_value=base_value
        )
        
        return value_matrix
    
    def _build_constraint_matrix(self):
        """Build the constraint matrix A of shape (3*K_priv + K_pub, 3*M)."""
        constraint_matrix = build_constraint_matrix(
            self.task_requirements, self.tasks, self.agents,
            self.private_resource_list, self.public_resource_list
        )
        return constraint_matrix
    
    def _run_solver(self, test_mode=False):
        """Run the ILP solver to find the optimal allocation."""
        value_vector = build_value_vector(self.value_matrix)
        
        optimal_allocation, max_reward = solve_task_allocation_with_state(
            self.vec_resource_state, self.constraint_matrix, value_vector,
            num_tasks=len(self.tasks), num_agents=len(self.agents)
        )
        
        if optimal_allocation is None or np.sum(optimal_allocation) == 0:
            if test_mode:
                raise ValueError("Error: No solution found in gt solver")
        
        return optimal_allocation, max_reward
    
    def agent_call_solver(self, total_resources, overall_preferences, natural_language=True):
        """
        Solver tool callable by agents during dialogue.

        Agents infer the global resource state and preferences through communication,
        then call this to compute an optimal allocation based on their beliefs.
        """
        agent_private_resources_obs = total_resources.get('agent_private_resources', {})
        public_resources_obs = total_resources.get('public_resources', {})
        
        if isinstance(overall_preferences, dict):
            print("Converting overall_preferences from dict to value_matrix...")
            print("Original overall_preferences dict:\n", overall_preferences)
            value_matrix = np.zeros((len(self.agents), len(self.tasks)))
            for agent_idx, agent_name in enumerate(self.agents):
                if agent_name in overall_preferences:
                    for task_idx, task_name in enumerate(self.tasks):
                        if task_name in overall_preferences[agent_name]:
                            value_matrix[agent_idx, task_idx] = overall_preferences[agent_name][task_name]
            overall_preferences = value_matrix
        print("Converted overall_preferences dict to value_matrix:\n", overall_preferences)
        
        vec_state = vectorize_task_allocation_state(
            agent_private_resources_obs, public_resources_obs,
            self.private_resource_list, self.public_resource_list
        )
        print("Vectorized resource state for agent_call_solver:\n", vec_state)
        
        value_vector = build_value_vector(overall_preferences)
        print("Built value vector for agent_call_solver:\n", value_vector)
        
        optimal_allocation, reward = solve_task_allocation_with_state(
            vec_state, self.constraint_matrix, value_vector,
            num_tasks=len(self.tasks), num_agents=len(self.agents)
        )
        
        if natural_language:
            allocation_dict = allocation_vector_to_dict(
                optimal_allocation, self.tasks, self.agents
            )
            return allocation_dict, reward
        else:
            return optimal_allocation, reward
    
    def agent_call_task_calculator(self, total_resources, natural_language=True):
        """
        Check which tasks can be assigned to which agents (based on resource constraints).

        Returns:
            available_allocations: {task: [feasible agents]}
            insufficient_allocation_info: {task: {agent: "Need more ..."}}
        """
        agent_private_resources_obs = total_resources.get('agent_private_resources', {})
        public_resources_obs = total_resources.get('public_resources', {})
        available_allocations = {}
        insufficient_allocation_info = {}
        
        for task_idx, task_name in enumerate(self.tasks):
            available_allocations[task_name] = []
            insufficient_allocation_info[task_name] = {}
            
            for agent_idx, agent_name in enumerate(self.agents):
                can_allocate = True
                missing_resources = []
                
                task_requirements = self.task_requirements[task_name]
                agent_requirements = task_requirements.get(agent_name, {})
                
                agent_private_resources = agent_private_resources_obs.get(agent_name, {})
                for res_name, res_needed in agent_requirements.items():
                    if res_name in self.private_resource_list:
                        res_available = agent_private_resources.get(res_name, 0)
                        if res_available < res_needed:
                            can_allocate = False
                            missing_resources.append(f"{res_name}: {res_needed - res_available}")
                
                # Support agent-specific public cost via agent_requirements
                public_requirements = task_requirements.get('public_resources', {})
                for res_name, res_needed in public_requirements.items():
                    if res_name in self.public_resource_list:
                        # if generator stored agent-specific public cost in agent_requirements, use it
                        effective_needed = agent_requirements.get(res_name, res_needed)
                        res_available = public_resources_obs.get(res_name, 0)
                        if res_available < effective_needed:
                            can_allocate = False
                            missing_resources.append(f"{res_name}: {effective_needed - res_available}")
                
                if can_allocate:
                    available_allocations[task_name].append(agent_name)
                else:
                    insufficient_allocation_info[task_name][agent_name] = f"Need more  {', '.join(missing_resources)}"
        
        return available_allocations, insufficient_allocation_info
    
    def evaluate_allocation(self, allocation):
        """Evaluate an allocation (dict or vector) and return its reward."""
        if isinstance(allocation, dict):
            allocation_vector = dict_to_allocation_vector(
                allocation, self.tasks, self.agents
            )
        else:
            allocation_vector = allocation
        
        value_vector = build_value_vector(self.value_matrix)
        
        reward = evaluate_allocation(
            allocation_vector, value_vector, self.constraint_matrix,
            self.vec_resource_state, num_tasks=len(self.tasks), num_agents=len(self.agents)
        )
        
        return reward
    
    def gt_representation(self):
        """Return a human-readable string of the game state (for debugging)."""
        repr_str = ''
        repr_str += 'Resource State (Ground Truth): \n'
        repr_str += json.dumps(self.game_resource_state, indent=4, ensure_ascii=False) + '\n'
        
        repr_str += '\nTask Requirements: \n'
        repr_str += json.dumps(self.task_requirements, indent=4, ensure_ascii=False) + '\n'
        
        repr_str += '\nValue Matrix (NxM): \n'
        value_dict = {}
        for agent_idx, agent_name in enumerate(self.agents):
            value_dict[agent_name] = {}
            for task_idx, task_name in enumerate(self.tasks):
                value_dict[agent_name][task_name] = float(self.value_matrix[agent_idx, task_idx])
        repr_str += json.dumps(value_dict, indent=4, ensure_ascii=False) + '\n'
        
        repr_str += '\nOptimal Allocation: \n'
        repr_str += json.dumps(self.best_allocation_dict, indent=4, ensure_ascii=False) + '\n'

        if self.task_values is not None:
            repr_str += '\nTask Values (sampled): \n'
            repr_str += json.dumps(self.task_values, indent=4, ensure_ascii=False) + '\n'
        
        repr_str += '\nMax Reward: \n'
        repr_str += str(self.max_reward) + '\n'
        
        repr_str += '\nAgent Observations (partial): \n'
        repr_str += 'agent_0 (Worker1) observation: \n'
        repr_str += json.dumps(self.agent_0_obs, indent=4, ensure_ascii=False) + '\n'
        repr_str += 'agent_1 (Worker2) observation: \n'
        repr_str += json.dumps(self.agent_1_obs, indent=4, ensure_ascii=False) + '\n'
        repr_str += 'agent_2 (Leader) observation: \n'
        repr_str += json.dumps(self.agent_2_obs, indent=4, ensure_ascii=False) + '\n'
        
        return repr_str


def select_task_allocation_games(
    games,
    top_k=60,
    priority_weights=None,
    min_tasks=3,
    reward_weight=0.01,
):
    """
    Select top_k games by a simple priority score.

    This mirrors the cooking pipeline's "select_games_with_initial_state_v1" idea,
    but adapted to task allocation:
    - prefer games where the solver can allocate more tasks (tasks are optional)
    - prefer higher resource utilization (avoid trivial instances)
    - slightly prefer higher max_reward to break ties
    """
    if priority_weights is None:
        # (w_p_done, w_utilization, w_reward)
        priority_weights = [1.0, 0.2, reward_weight]
    if len(priority_weights) != 3:
        raise ValueError("priority_weights must have length 3: [w_p_done, w_utilization, w_reward]")

    w_p_done, w_util, w_reward = [float(x) for x in priority_weights]

    priorities = []
    for game in games:
        n_total = len(game.tasks)
        # tasks are optional; best_allocation_dict contains assigned tasks
        n_done = len(game.best_allocation_dict)
        p_done = (n_done / n_total) if n_total > 0 else 0.0

        # compute resource utilization based on solver output
        util = 0.0
        if game.best_allocation is not None and np.sum(game.best_allocation) > 0:
            usage = game.constraint_matrix @ game.best_allocation
            cap = game.vec_resource_state
            mask = cap > 1e-6
            if np.any(mask):
                util = float(np.mean((usage[mask] / cap[mask]).clip(0.0, 1.0)))

        # filter out degenerate instances
        if n_done < int(min_tasks) or game.max_reward <= 0.0:
            priority = -1e9
        else:
            priority = w_p_done * p_done + w_util * util + w_reward * float(game.max_reward)
        priorities.append(priority)

    priorities = np.array(priorities, dtype=np.float32)
    selected_idx = np.argsort(-priorities)[: int(top_k)]
    return [games[i] for i in selected_idx.tolist()]

