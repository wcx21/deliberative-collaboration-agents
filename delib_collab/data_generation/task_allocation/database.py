"""
Task allocation database utilities (database_1225)

This file combines:
- database loading
- sampling utilities
- format conversion to TaskAllocationGame inputs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from delib_collab.paths import TASK_ALLOCATION_SOURCE_DATA_DIR


def _resolve_database_root(database_root: str) -> str:
    if os.path.exists(database_root):
        return database_root
    candidates = [
        os.path.join(str(TASK_ALLOCATION_SOURCE_DATA_DIR), database_root),
        os.path.join(str(TASK_ALLOCATION_SOURCE_DATA_DIR), os.path.basename(database_root)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    return [ln for ln in lines if ln]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class DomainDatabase:
    domain_name: str
    domain_path: str
    resources: List[str]
    private_resources: List[str]
    public_resources: List[str]
    tasks: List[str]
    task_information: Dict[str, Any]
    agent_pool: List[Dict[str, Any]]


def get_all_domains(database_root: str) -> List[str]:
    """
    List domain folders under database_root, ignoring files.
    """
    database_root = _resolve_database_root(database_root)
    if not os.path.exists(database_root):
        raise FileNotFoundError(f"database_root does not exist: {database_root}")

    domains: List[str] = []
    for name in os.listdir(database_root):
        if name.startswith("__"):
            continue
        p = os.path.join(database_root, name)
        if os.path.isdir(p):
            domains.append(name)
    domains.sort()
    return domains


def load_database_domain(database_root: str, domain_name: str) -> DomainDatabase:
    """
    Load one domain folder in database_1225 format.
    """
    database_root = _resolve_database_root(database_root)
    domain_path = os.path.join(database_root, domain_name)
    if not os.path.isdir(domain_path):
        raise FileNotFoundError(f"domain folder does not exist: {domain_path}")

    resources = _read_lines(os.path.join(domain_path, "resources.txt"))
    private_resources = _read_lines(os.path.join(domain_path, "private_resources.txt"))
    public_resources = _read_lines(os.path.join(domain_path, "public_resources.txt"))
    tasks = _read_lines(os.path.join(domain_path, "tasks.txt"))
    task_information = _read_json(os.path.join(domain_path, "task_information.json"))
    agents = _read_json(os.path.join(domain_path, "agents.json"))
    agent_pool = agents.get("agent_pool", [])
    if not isinstance(agent_pool, list):
        raise ValueError(f"agents.json agent_pool should be a list in {domain_path}")

    return DomainDatabase(
        domain_name=domain_name,
        domain_path=domain_path,
        resources=resources,
        private_resources=private_resources,
        public_resources=public_resources,
        tasks=tasks,
        task_information=task_information,
        agent_pool=agent_pool,
    )


def _rng(seed: Optional[int], salt: int = 0) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    # simple deterministic derivation to avoid collisions
    return np.random.default_rng(int(seed) + int(salt) * 1000003)


def sample_tasks_from_domain(task_list: Sequence[str], n_tasks: int = 10, seed: Optional[int] = None) -> List[str]:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if len(task_list) < n_tasks:
        raise ValueError(f"Not enough tasks to sample: have {len(task_list)}, need {n_tasks}")
    rng = _rng(seed)
    idx = rng.choice(len(task_list), size=n_tasks, replace=False)
    return [task_list[i] for i in idx.tolist()]


def sample_personas_from_pool(
    agent_pool: Sequence[Dict[str, Any]],
    n_workers: int = 2,
    n_leaders: int = 1,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns: (leader_persona, worker_personas)
    """
    if n_leaders != 1:
        raise ValueError("This repo assumes exactly 1 leader persona per game.")
    if n_workers != 2:
        raise ValueError("This repo assumes exactly 2 worker personas per game.")

    rng = _rng(seed)
    leaders = [p for p in agent_pool if "leader" in p.get("possible_roles", [])]
    workers = [p for p in agent_pool if "worker" in p.get("possible_roles", [])]
    if len(leaders) < 1:
        raise ValueError("No leader personas available in agent_pool.")
    if len(workers) < 2:
        raise ValueError("Not enough worker personas available in agent_pool.")

    leader = leaders[int(rng.integers(low=0, high=len(leaders)))]

    # ensure workers are distinct from leader (by persona_id) and distinct from each other
    leader_id = leader.get("persona_id")
    eligible_workers = [p for p in workers if p.get("persona_id") != leader_id]
    if len(eligible_workers) < 2:
        raise ValueError("Not enough distinct worker personas after excluding leader.")

    worker_idx = rng.choice(len(eligible_workers), size=2, replace=False)
    worker_personas = [eligible_workers[i] for i in worker_idx.tolist()]
    return leader, worker_personas


def _trunc_pos(x: float, eps: float = 1e-6) -> float:
    return float(max(eps, x))


def _trunc_nonneg(x: float) -> float:
    return float(max(0.0, x))


def _round_to_0_5(x: float) -> float:
    """Round value to nearest multiple of 5."""
    return float(round(x * 5.0) / 5.0)


def _round_to_0_1(x: float) -> float:
    """Round value to nearest multiple of 0.1."""
    return float(round(x * 10.0) / 10.0)


def sample_task_values(
    task_information: Dict[str, Any],
    selected_tasks: Sequence[str],
    seed: Optional[int] = None,
) -> Dict[str, float]:
    rng = _rng(seed)
    sampled: Dict[str, float] = {}
    for t in selected_tasks:
        info = task_information.get(t)
        if info is None:
            raise KeyError(f"Task not found in task_information.json: {t}")
        base_value = info.get("base_value", {})
        mu = float(base_value.get("mean", 1.0))
        sigma = float(base_value.get("std", 0.0))
        v = float(rng.normal(loc=mu, scale=sigma))
        v = _trunc_pos(v)
        sampled[t] = _round_to_0_5(v)
    return sampled


def sample_agent_private_capacities(
    personas: Sequence[Dict[str, Any]],
    private_resource_list: Sequence[str],
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Returns {persona_id: {private_resource: sampled_capacity}}
    """
    rng = _rng(seed)
    out: Dict[str, Dict[str, float]] = {}
    for persona in personas:
        pid = persona.get("persona_id")
        if not pid:
            raise ValueError("Persona missing persona_id")
        pr = persona.get("private_resources", {})
        cap: Dict[str, float] = {}
        for r in private_resource_list:
            dist = pr.get(r)
            if dist is None:
                raise KeyError(f"Persona {pid} missing private resource distribution for '{r}'")
            mu = float(dist.get("mean", 0.0))
            sigma = float(dist.get("std", 0.0))
            v = float(rng.normal(loc=mu, scale=sigma))
            v = _trunc_nonneg(v)
            cap[r] = _round_to_0_5(v)
        out[str(pid)] = cap
    return out


def sample_public_resource_capacities(
    task_information: Dict[str, Any],
    selected_tasks: Sequence[str],
    public_resource_list: Sequence[str],
    margin_factor: float = 1.5,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Sample team-level public resource capacities.

    Simple heuristic: capacity ~= ceil(margin_factor * sum_baseline_requirements_over_tasks)
    """
    if margin_factor <= 0:
        raise ValueError("margin_factor must be positive")
    rng = _rng(seed)
    baseline_sum: Dict[str, float] = {r: 0.0 for r in public_resource_list}
    for t in task_information:
        info = task_information[t]
        pub_req = info.get("public_resource_requirements", {}) or {}
        for r in public_resource_list:
            baseline_sum[r] += float(pub_req.get(r, 0)) * len(selected_tasks) / len(task_information)

    capacities: Dict[str, int] = {}
    for r in public_resource_list:
        base = baseline_sum[r] * margin_factor
        # add small noise to avoid identical instances across seeds
        jitter = float(rng.uniform(0.5, 1.5))
        cap = float(base * jitter)
        cap = _round_to_0_5(cap)
        capacities[r] = max(0, int(cap))
    return capacities


def convert_database_to_game_format(
    selected_tasks: Sequence[str],
    task_information: Dict[str, Any],
    leader_persona: Dict[str, Any],
    worker_personas: Sequence[Dict[str, Any]],
    sampled_task_values: Dict[str, float],
    sampled_private_capacities: Dict[str, Dict[str, float]],
    sampled_public_capacities: Dict[str, int],
    resources: Sequence[str],
    private_resource_list: Sequence[str],
    public_resource_list: Sequence[str],
    use_public_resource_multipliers: bool = False,
) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """
    Convert database_1225 schema into TaskAllocationGame inputs.

    Returns:
      - tasks (list)
      - resources (list)
      - task_requirements (dict)
      - agents_config (dict)
      - resource_state (dict)
      - task_values (dict task->value)
    """
    if len(worker_personas) != 2:
        raise ValueError("Expected 2 worker personas.")

    agent_personas = {
        "agent_0": worker_personas[0],
        "agent_1": worker_personas[1],
        "agent_2": leader_persona,
    }

    # agents_config: includes efficiency and a *sampled* private_resources capacity dict (like demo_01)
    agents_config: Dict[str, Any] = {}
    for agent_name, persona in agent_personas.items():
        pid = str(persona.get("persona_id"))
        # Round efficiency values to 0.1
        efficiency_raw = persona.get("efficiency", {})
        efficiency_rounded = {k: _round_to_0_1(float(v)) for k, v in efficiency_raw.items()}
        agents_config[agent_name] = {
            "name": persona.get("name", agent_name),
            "role": "Leader" if agent_name == "agent_2" else "Worker",
            "description": persona.get("description", ""),
            "private_resources": sampled_private_capacities[pid],
            "efficiency": efficiency_rounded,
            "public_resource_cost_multipliers": persona.get("public_resource_cost_multipliers", {}),
        }

    # task_requirements: per-task includes per-agent private reqs and (baseline) public reqs.
    # To model public multipliers, we additionally store per-agent public costs *inside* each agent dict.
    task_requirements: Dict[str, Any] = {}
    for t in selected_tasks:
        info = task_information[t]
        base_priv = info.get("private_resource_requirements", {}) or {}
        base_pub = info.get("public_resource_requirements", {}) or {}

        # Round public_resources baseline costs to 5
        public_resources_rounded = {r: _round_to_0_5(float(base_pub.get(r, 0))) for r in public_resource_list}
        task_requirements[t] = {"public_resources": public_resources_rounded}

        for agent_name, persona in agent_personas.items():
            agent_req: Dict[str, float] = {}

            # private costs (agent-agnostic baseline) - round to 5
            for r in private_resource_list:
                agent_req[r] = _round_to_0_5(float(base_priv.get(r, 0)))

            # public costs: apply multipliers only if enabled, otherwise use baseline
            if use_public_resource_multipliers:
                mults = persona.get("public_resource_cost_multipliers", {}) or {}
                for r in public_resource_list:
                    base_cost = float(base_pub.get(r, 0))
                    m = float(mults.get(r, 1.0))
                    cost = base_cost * m
                    agent_req[r] = _round_to_0_5(cost)
            else:
                # use baseline public resource costs for all agents (no multipliers)
                for r in public_resource_list:
                    base_cost = float(base_pub.get(r, 0))
                    agent_req[r] = _round_to_0_5(base_cost)

            task_requirements[t][agent_name] = agent_req

    resource_state = {
        "agent_private_resources": {
            agent_name: agents_config[agent_name]["private_resources"] for agent_name in ["agent_0", "agent_1", "agent_2"]
        },
        "public_resources": {r: int(sampled_public_capacities.get(r, 0)) for r in public_resource_list},
    }

    # task values aligned with selected tasks
    task_values = {t: float(sampled_task_values[t]) for t in selected_tasks}

    return list(selected_tasks), list(resources), task_requirements, agents_config, resource_state, task_values
