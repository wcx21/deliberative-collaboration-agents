"""
Task allocation game generation script (from database).

Analogue of the cooking game generation script, but for the
task-allocation format (`TaskAllocationGame`).

This script:
- samples tasks (~10) from a domain
- samples personas (1 leader + 2 workers)
- samples task values and private/public capacities
- converts database schema into TaskAllocationGame inputs
- creates many candidate games, selects top ones, and saves them as pickle files
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=""):
        if desc:
            print(f"{desc}...")
        return iterable

from delib_collab.data_generation.task_allocation.io import save_games
from delib_collab.data_generation.task_allocation.game import TaskAllocationGame, level1_resource_partition, select_task_allocation_games
from delib_collab.data_generation.task_allocation.database import (
    get_all_domains,
    load_database_domain,
    sample_tasks_from_domain,
    sample_personas_from_pool,
    sample_task_values,
    sample_agent_private_capacities,
    sample_public_resource_capacities,
    convert_database_to_game_format,
)


def gen_task_allocation_games(
    database_root: str = "database",
    output_dir: str = "task_allocation_games_20250127",
    n_seed: int = 2000,
    n_games: int = 60,
    n_tasks_per_game: int = 10,
    domains: Optional[Sequence[str]] = None,
    seed: Optional[int] = 42,
    obs_min_split_range: float = 0.4,
    leader_public_ratio: float = 0.6,
    public_resource_margin: float = 0.8,
    select_priority_weights=None,
    min_tasks: int = 3,
    use_public_resource_multipliers: bool = False,
):
    """
    Generate task allocation games and save them into data/task_allocation/games/{output_dir}/games/level_1_and_2/.

    Notes:
    - Tasks are optional (solver allows leaving tasks unassigned when infeasible).
    - Public resource costs: if use_public_resource_multipliers=True, costs are agent-dependent via persona multipliers.
      If False (default), all agents use the same baseline public resource costs.
    """
    if n_seed <= 0:
        raise ValueError("n_seed must be positive")
    if n_games <= 0:
        raise ValueError("n_games must be positive")
    if n_tasks_per_game <= 0:
        raise ValueError("n_tasks_per_game must be positive")

    if domains is None:
        domains = get_all_domains(database_root)
    domains = list(domains)
    if len(domains) == 0:
        raise ValueError("No domains to generate from.")

    # split n_games across domains
    base = n_games // len(domains)
    rem = n_games % len(domains)
    n_games_per_domain = [base + (1 if i < rem else 0) for i in range(len(domains))]

    all_selected: List[TaskAllocationGame] = []

    for domain_idx, domain_name in enumerate(domains):
        print("=" * 60)
        print(f"Generating domain: {domain_name}")
        print("=" * 60)

        db = load_database_domain(database_root, domain_name)

        # generate candidate games
        games: List[TaskAllocationGame] = []
        for i in tqdm(range(n_seed), desc=f"Sampling candidates ({domain_name})"):
            s_base = None if seed is None else int(seed) + domain_idx * 100000 + i

            selected_tasks = sample_tasks_from_domain(db.tasks, n_tasks=n_tasks_per_game, seed=s_base)
            leader, workers = sample_personas_from_pool(db.agent_pool, seed=None if s_base is None else s_base + 1)
            task_values = sample_task_values(db.task_information, selected_tasks, seed=None if s_base is None else s_base + 2)
            priv_caps = sample_agent_private_capacities(
                personas=[leader, workers[0], workers[1]],
                private_resource_list=db.private_resources,
                seed=None if s_base is None else s_base + 3,
            )
            pub_caps = sample_public_resource_capacities(
                db.task_information,
                selected_tasks,
                db.public_resources,
                margin_factor=public_resource_margin,
                seed=None if s_base is None else s_base + 4,
            )

            tasks, resources, task_requirements, agents_config, resource_state, task_values_out = convert_database_to_game_format(
                selected_tasks=selected_tasks,
                task_information=db.task_information,
                leader_persona=leader,
                worker_personas=workers,
                sampled_task_values=task_values,
                sampled_private_capacities=priv_caps,
                sampled_public_capacities=pub_caps,
                resources=db.resources,
                private_resource_list=db.private_resources,
                public_resource_list=db.public_resources,
                use_public_resource_multipliers=use_public_resource_multipliers,
            )

            # partial observation for level 1
            agent_0_obs, agent_1_obs, agent_2_obs = level1_resource_partition(
                resource_state["agent_private_resources"],
                resource_state["public_resources"],
                min_split_range=obs_min_split_range,
                leader_public_ratio=leader_public_ratio,
            )

            game = TaskAllocationGame(
                tasks=tasks,
                resources=resources,
                task_requirements=task_requirements,
                agents_config=agents_config,
                resource_state=resource_state,
                partial_observations=(agent_0_obs, agent_1_obs, agent_2_obs),
                task_values=task_values_out,
            )

            # attach metadata for debugging
            game.domain_name = domain_name  # type: ignore[attr-defined]
            game.persona_ids = {
                "agent_0": workers[0].get("persona_id"),
                "agent_1": workers[1].get("persona_id"),
                "agent_2": leader.get("persona_id"),
            }  # type: ignore[attr-defined]

            games.append(game)

        # select top games for this domain
        k = n_games_per_domain[domain_idx]
        selected = select_task_allocation_games(
            games,
            top_k=k,
            priority_weights=select_priority_weights,
            min_tasks=min_tasks,
        )
        all_selected.extend(selected)
        print(f"Selected {len(selected)} games for domain {domain_name}.")

    print("=" * 60)
    print(f"Saving total {len(all_selected)} games to dataset: {output_dir}")
    print("=" * 60)
    save_games(output_dir, all_selected, game_batch_name="level_1_and_2", overwrite=True)

    return all_selected


if __name__ == "__main__":
    # Example: generate a small dataset (edit parameters as needed)
    gen_task_allocation_games(
        database_root="database",
        output_dir="task_allocation_games",
        n_seed=1000,
        n_games=60,
        n_tasks_per_game=10,
        seed=42,
        min_tasks=3,
        select_priority_weights=[0.1, 1, 0],
        use_public_resource_multipliers=False
    )

