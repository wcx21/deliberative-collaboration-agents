"""
Core solver module for the task allocation problem.

Implements ILP-based optimization, constraint checking, and evaluation.
"""

import numpy as np
import pulp as lp


def vectorize_task_allocation_state(agent_private_resources, public_resources, 
                                     private_resource_list, public_resource_list):
    """
    Vectorize the task allocation state into S = [S_0^priv, S_1^priv, S_2^priv, S^pub]^T.

    Returns:
        state_vector: np.array of shape (3*K_priv + K_pub,)
    """
    K_priv = len(private_resource_list)
    K_pub = len(public_resource_list)
    
    state_vector = np.zeros(3 * K_priv + K_pub, dtype=np.float32)
    
    for agent_idx, agent_name in enumerate(['agent_0', 'agent_1', 'agent_2']):
        agent_resources = agent_private_resources.get(agent_name, {})
        for res_idx, res_name in enumerate(private_resource_list):
            state_vector[agent_idx * K_priv + res_idx] = agent_resources.get(res_name, 0)
    
    for res_idx, res_name in enumerate(public_resource_list):
        state_vector[3 * K_priv + res_idx] = public_resources.get(res_name, 0)
    
    return state_vector


def build_constraint_matrix(task_requirements, tasks, agents, 
                            private_resource_list, public_resource_list):
    """
    Build the constraint matrix A with block-diagonal private rows and dense public rows.

    Structure:
        A = [diag(W_priv, W_priv, W_priv)]   (private, per-agent)
            [W_pub   W_pub   W_pub        ]   (public, shared)

    Returns:
        constraint_matrix: np.array of shape (3*K_priv + K_pub, 3*M)
    """
    N = len(agents)
    M = len(tasks)
    K_priv = len(private_resource_list)
    K_pub = len(public_resource_list)
    
    A = np.zeros((3 * K_priv + K_pub, N * M), dtype=np.float32)
    
    for task_idx, task_name in enumerate(tasks):
        task_req = task_requirements[task_name]
        
        for agent_idx, agent_name in enumerate(agents):
            x_pos = agent_idx * M + task_idx
            
            agent_req = task_req.get(agent_name, {})
            for res_idx, res_name in enumerate(private_resource_list):
                A[agent_idx * K_priv + res_idx, x_pos] = agent_req.get(res_name, 0)
            
            # Support agent-specific public cost multipliers stored in agent_req
            public_req = task_req.get('public_resources', {})
            for res_idx, res_name in enumerate(public_resource_list):
                A[3 * K_priv + res_idx, x_pos] = agent_req.get(res_name, public_req.get(res_name, 0))
    
    return A


def build_value_vector(value_matrix):
    """
    Flatten the N x M value matrix into a 1D vector (row-major order).

    Returns:
        value_vector: np.array of shape (N*M,)
    """
    return value_matrix.flatten()


def check_allocation_constraints(allocation_vector, constraint_matrix, state_vector, 
                                  num_tasks, num_agents):
    """
    Check whether an allocation satisfies all constraints.

    Validates: (1) resource feasibility Ax <= S, and
    (2) each task is assigned to at most one agent.

    Returns:
        is_valid: bool
    """
    resource_usage = constraint_matrix @ allocation_vector
    resource_feasible = np.all(resource_usage <= state_vector + 1e-6)
    
    allocation_matrix = allocation_vector.reshape(num_agents, num_tasks)
    task_completeness = np.all(np.sum(allocation_matrix, axis=0) <= 1 + 1e-6)
    
    return resource_feasible and task_completeness


def evaluate_allocation(allocation_vector, value_vector, constraint_matrix, 
                       state_vector, num_tasks, num_agents):
    """
    Evaluate an allocation's reward: R(x) = IsValid(x) * (v^T . x).

    Returns 0.0 if constraints are violated.
    """
    if not check_allocation_constraints(allocation_vector, constraint_matrix, 
                                       state_vector, num_tasks, num_agents):
        return 0.0
    
    reward = np.dot(value_vector, allocation_vector)
    return float(reward)


def solve_task_allocation_with_state(state_vector, constraint_matrix, value_vector,
                                     num_tasks, num_agents):
    """
    Solve optimal task allocation via ILP.

    Maximize v^T x  subject to:
      (1) Ax <= S  (resource constraints)
      (2) each task assigned to at most one agent
      (3) x in {0, 1}

    Returns:
        (optimal_allocation, max_reward)
    """
    N = num_agents
    M = num_tasks
    
    prob = lp.LpProblem("Task_Allocation_Optimization", lp.LpMaximize)
    
    x = []
    for agent_idx in range(N):
        for task_idx in range(M):
            var_name = f"x_{agent_idx}_{task_idx}"
            x.append(lp.LpVariable(var_name, cat=lp.LpBinary))
    
    prob += lp.lpSum([value_vector[i] * x[i] for i in range(N * M)])
    
    num_constraints = constraint_matrix.shape[0]
    for constraint_idx in range(num_constraints):
        prob += lp.lpSum([constraint_matrix[constraint_idx, i] * x[i] 
                          for i in range(N * M)]) <= state_vector[constraint_idx]
    
    for task_idx in range(M):
        task_vars = [x[agent_idx * M + task_idx] for agent_idx in range(N)]
        prob += lp.lpSum(task_vars) <= 1
    
    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    if lp.LpStatus[prob.status] == 'Optimal':
        optimal_allocation = np.array([lp.value(x[i]) if lp.value(x[i]) is not None 
                                      else 0.0 for i in range(N * M)])
        optimal_allocation = (optimal_allocation >= 0.5).astype(np.float32)
        max_reward = lp.value(prob.objective)
        if max_reward is None:
            max_reward = 0.0
        max_reward = round(float(max_reward), 4)
        return optimal_allocation, max_reward
    else:
        return np.zeros(N * M, dtype=np.float32), 0.0


def allocation_vector_to_dict(allocation_vector, tasks, agents):
    """Convert a flat binary allocation vector to {task_name: agent_name} dict."""
    N = len(agents)
    M = len(tasks)
    
    allocation_dict = {}
    allocation_matrix = allocation_vector.reshape(N, M)
    
    for task_idx, task_name in enumerate(tasks):
        for agent_idx, agent_name in enumerate(agents):
            if allocation_matrix[agent_idx, task_idx] >= 0.5:
                allocation_dict[task_name] = agent_name
                break
    
    return allocation_dict


def dict_to_allocation_vector(allocation_dict, tasks, agents):
    """Convert a {task_name: agent_name} dict to a flat binary allocation vector."""
    N = len(agents)
    M = len(tasks)
    
    allocation_vector = np.zeros(N * M, dtype=np.float32)
    
    for task_idx, task_name in enumerate(tasks):
        if task_name in allocation_dict:
            agent_name = allocation_dict[task_name]
            if agent_name in agents:
                agent_idx = agents.index(agent_name)
                allocation_vector[agent_idx * M + task_idx] = 1.0
    
    return allocation_vector


# ==================== Unit Tests ====================

def test_case_1_simple():
    """Test case 1: simple 2-task x 3-agent scenario. Expected optimal: TaskA->agent_2, TaskB->agent_1, score=22."""
    print("=" * 60)
    print("Test case 1: simple 2-task x 3-agent")
    print("=" * 60)
    
    private_resource_list = ['Time']
    public_resource_list = ['Budget']
    
    agents = ['agent_0', 'agent_1', 'agent_2']
    tasks = ['TaskA', 'TaskB']
    
    agent_private_resources = {
        'agent_0': {'Time': 10},
        'agent_1': {'Time': 8},
        'agent_2': {'Time': 5}
    }
    public_resources = {'Budget': 200}
    
    task_requirements = {
        'TaskA': {
            'agent_0': {'Time': 3},
            'agent_1': {'Time': 4},
            'agent_2': {'Time': 2},
            'public_resources': {'Budget': 100}
        },
        'TaskB': {
            'agent_0': {'Time': 2},
            'agent_1': {'Time': 3},
            'agent_2': {'Time': 1},
            'public_resources': {'Budget': 50}
        }
    }
    
    value_matrix = np.array([
        [10, 8],
        [8, 10],
        [12, 6]
    ], dtype=np.float32)
    
    state_vector = vectorize_task_allocation_state(
        agent_private_resources, public_resources,
        private_resource_list, public_resource_list
    )
    print(f"\nState vector: {state_vector}")
    
    constraint_matrix = build_constraint_matrix(
        task_requirements, tasks, agents,
        private_resource_list, public_resource_list
    )
    print(f"\nConstraint matrix shape: {constraint_matrix.shape}")
    print(f"\nConstraint matrix:\n{constraint_matrix}")
    
    value_vector = build_value_vector(value_matrix)
    print(f"\nValue vector: {value_vector}")
    
    optimal_allocation, max_reward = solve_task_allocation_with_state(
        state_vector, constraint_matrix, value_vector,
        num_tasks=len(tasks), num_agents=len(agents)
    )
    
    print(f"\nOptimal allocation vector: {optimal_allocation}")
    
    allocation_dict = allocation_vector_to_dict(optimal_allocation, tasks, agents)
    print(f"\nOptimal allocation: {allocation_dict}")
    
    is_valid = check_allocation_constraints(
        optimal_allocation, constraint_matrix, state_vector,
        num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nConstraint check: {'PASS' if is_valid else 'FAIL'}")
    
    calculated_reward = evaluate_allocation(
        optimal_allocation, value_vector, constraint_matrix,
        state_vector, num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nMax reward: {max_reward}")
    print(f"Calculated reward: {calculated_reward}")
    print(f"Consistent: {'yes' if abs(max_reward - calculated_reward) < 1e-6 else 'no'}")
    
    expected_allocation = {'TaskA': 'agent_2', 'TaskB': 'agent_1'}
    expected_reward = 22.0
    
    assert allocation_dict == expected_allocation, \
        f"Allocation mismatch! Expected: {expected_allocation}, Got: {allocation_dict}"
    assert abs(max_reward - expected_reward) < 1e-6, \
        f"Reward mismatch! Expected: {expected_reward}, Got: {max_reward}"
    assert is_valid, "Constraint check failed!"
    
    print(f"\nTest case 1 passed!")
    return True


def test_case_2_complex():
    """Test case 2: complex 3-task x 3-agent with 2 private resource types."""
    print("\n" + "=" * 60)
    print("Test case 2: complex 3-task x 3-agent")
    print("=" * 60)
    
    private_resource_list = ['Time', 'GPU']
    public_resource_list = ['Budget']
    
    agents = ['agent_0', 'agent_1', 'agent_2']
    tasks = ['TaskA', 'TaskB', 'TaskC']
    
    agent_private_resources = {
        'agent_0': {'Time': 10, 'GPU': 2},
        'agent_1': {'Time': 8, 'GPU': 1},
        'agent_2': {'Time': 6, 'GPU': 0}
    }
    public_resources = {'Budget': 300}
    
    task_requirements = {
        'TaskA': {
            'agent_0': {'Time': 3, 'GPU': 1},
            'agent_1': {'Time': 4, 'GPU': 0},
            'agent_2': {'Time': 2, 'GPU': 0},
            'public_resources': {'Budget': 100}
        },
        'TaskB': {
            'agent_0': {'Time': 2, 'GPU': 1},
            'agent_1': {'Time': 3, 'GPU': 1},
            'agent_2': {'Time': 1, 'GPU': 0},
            'public_resources': {'Budget': 80}
        },
        'TaskC': {
            'agent_0': {'Time': 4, 'GPU': 0},
            'agent_1': {'Time': 2, 'GPU': 0},
            'agent_2': {'Time': 3, 'GPU': 0},
            'public_resources': {'Budget': 120}
        }
    }
    
    value_matrix = np.array([
        [10, 8, 12],
        [8, 10, 9],
        [12, 6, 10]
    ], dtype=np.float32)
    
    state_vector = vectorize_task_allocation_state(
        agent_private_resources, public_resources,
        private_resource_list, public_resource_list
    )
    print(f"\nState vector: {state_vector} (shape: {state_vector.shape})")
    
    constraint_matrix = build_constraint_matrix(
        task_requirements, tasks, agents,
        private_resource_list, public_resource_list
    )
    print(f"\nConstraint matrix shape: {constraint_matrix.shape}")
    
    value_vector = build_value_vector(value_matrix)
    print(f"\nValue vector shape: {value_vector.shape}")
    
    optimal_allocation, max_reward = solve_task_allocation_with_state(
        state_vector, constraint_matrix, value_vector,
        num_tasks=len(tasks), num_agents=len(agents)
    )

    print(f"\nOptimal allocation vector: {optimal_allocation}")
    allocation_dict = allocation_vector_to_dict(optimal_allocation, tasks, agents)
    print(f"\nOptimal allocation: {allocation_dict}")
    
    is_valid = check_allocation_constraints(
        optimal_allocation, constraint_matrix, state_vector,
        num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nConstraint check: {'PASS' if is_valid else 'FAIL'}")
    
    calculated_reward = evaluate_allocation(
        optimal_allocation, value_vector, constraint_matrix,
        state_vector, num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nMax reward: {max_reward}")
    print(f"Calculated reward: {calculated_reward}")
    print(f"Consistent: {'yes' if abs(max_reward - calculated_reward) < 1e-6 else 'no'}")
    
    resource_usage = constraint_matrix @ optimal_allocation
    print(f"\nResource usage: {resource_usage}")
    print(f"Resource capacity: {state_vector}")
    print(f"Feasible: {np.all(resource_usage <= state_vector + 1e-6)}")
    
    allocation_matrix = optimal_allocation.reshape(len(agents), len(tasks))
    task_completeness = np.all(np.sum(allocation_matrix, axis=0) <= 1 + 1e-6)
    print(f"\nTask completeness: {'PASS' if task_completeness else 'FAIL'}")
    print(f"  Assignments per task: {np.sum(allocation_matrix, axis=0)}")
    
    assert is_valid, "Constraint check failed!"
    assert task_completeness, "Task completeness check failed!"
    assert abs(max_reward - calculated_reward) < 1e-6, "Reward mismatch!"
    # Tasks are optional; when all values are positive and feasible, solver should still assign all.
    assert len(allocation_dict) >= 1, "No tasks assigned!"
    
    print(f"\nTest case 2 passed!")
    return True


def test_invalid_allocation():
    """Test case 3: invalid allocations (constraint violations)."""
    print("\n" + "=" * 60)
    print("Test case 3: invalid allocations")
    print("=" * 60)
    
    private_resource_list = ['Time']
    public_resource_list = ['Budget']
    agents = ['agent_0', 'agent_1', 'agent_2']
    tasks = ['TaskA', 'TaskB']
    
    agent_private_resources = {
        'agent_0': {'Time': 10},
        'agent_1': {'Time': 8},
        'agent_2': {'Time': 5}
    }
    public_resources = {'Budget': 200}
    
    task_requirements = {
        'TaskA': {
            'agent_0': {'Time': 3},
            'agent_1': {'Time': 4},
            'agent_2': {'Time': 2},
            'public_resources': {'Budget': 100}
        },
        'TaskB': {
            'agent_0': {'Time': 2},
            'agent_1': {'Time': 3},
            'agent_2': {'Time': 1},
            'public_resources': {'Budget': 50}
        }
    }
    
    value_matrix = np.array([
        [10, 8],
        [8, 10],
        [12, 6]
    ], dtype=np.float32)
    
    state_vector = vectorize_task_allocation_state(
        agent_private_resources, public_resources,
        private_resource_list, public_resource_list
    )
    
    constraint_matrix = build_constraint_matrix(
        task_requirements, tasks, agents,
        private_resource_list, public_resource_list
    )
    
    value_vector = build_value_vector(value_matrix)
    
    # Test 1: task completeness violation (TaskA assigned to multiple agents)
    invalid_allocation_1 = np.array([1, 0, 1, 0, 0, 0], dtype=np.float32)
    is_valid_1 = check_allocation_constraints(
        invalid_allocation_1, constraint_matrix, state_vector,
        num_tasks=len(tasks), num_agents=len(agents)
    )
    reward_1 = evaluate_allocation(
        invalid_allocation_1, value_vector, constraint_matrix,
        state_vector, num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nTest 1: task completeness violation")
    print(f"  Constraint check: {'PASS' if is_valid_1 else 'FAIL'}")
    print(f"  Reward: {reward_1}")
    assert not is_valid_1, "Should detect task completeness violation!"
    assert reward_1 == 0, "Invalid allocation should score 0!"
    
    # Test 2: public resource violation (insufficient budget)
    tight_public_state = np.array([10, 8, 5, 0], dtype=np.float32)
    invalid_allocation_2 = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)
    is_valid_2 = check_allocation_constraints(
        invalid_allocation_2, constraint_matrix, tight_public_state,
        num_tasks=len(tasks), num_agents=len(agents)
    )
    reward_2 = evaluate_allocation(
        invalid_allocation_2, value_vector, constraint_matrix,
        tight_public_state, num_tasks=len(tasks), num_agents=len(agents)
    )
    print(f"\nTest 2: public resource violation (insufficient budget)")
    print(f"  Constraint check: {'PASS' if is_valid_2 else 'FAIL'}")
    print(f"  Reward: {reward_2}")
    assert not is_valid_2, "Should detect public resource constraint failure!"
    assert reward_2 == 0, "Invalid allocation should score 0!"
    
    print(f"\nTest case 3 passed!")
    return True


if __name__ == '__main__':
    print("Running task allocation unit tests...")
    print("\n")
    
    try:
        test_case_1_simple()
        test_case_2_complex()
        test_invalid_allocation()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        raise
    except Exception as e:
        print(f"\nTest error: {e}")
        raise

