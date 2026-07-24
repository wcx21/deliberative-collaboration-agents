import os, sys
import argparse
import multiprocessing
import subprocess
from multiprocessing import Manager
from delib_collab.paths import PROJECT_ROOT

root_dir = str(PROJECT_ROOT)
agent_dir = os.path.join(root_dir, 'delib_collab', 'agents')

parser = argparse.ArgumentParser(description='Process all with dynamic task allocation')
parser.add_argument('--scenario', type=str, default='cook', choices=['cook', 'task_allo'], 
                    help='Scenario type: cook or task_allo')
parser.add_argument('--oracle', default=False, action='store_true', 
                    help='Enable oracle agent mode (works with both cook and task_allo scenarios)')
parser.add_argument('--exp_levels', type=str, default=['1'], nargs='+', help='Experiment levels')
parser.add_argument('-n', '--num_process', type=int, default=None,
                    help='Total number of concurrent processes (if specified, will calculate n_procs_per_model based on number of models)')
parser.add_argument('--n_procs_per_model', type=int, default=None, 
                    help='Number of concurrent processes per model (if not specified and num_process is specified, will be calculated)')
parser.add_argument('-g', '--game_folder', type=str, default='test_games', help='game folder')
parser.add_argument('-s', '--start', type=int, default=None, help='start of games')
parser.add_argument('-e', '--end', type=int, default=None, help='end of games')
parser.add_argument('--max_round', type=int, default=6, help='')
parser.add_argument('--max_character', type=int, default=3000, help='')
parser.add_argument('-m', '--models', type=str, default=["model-name"], nargs='+', help='LLMs')
parser.add_argument('--exp_name', type=str, default='tmp', help='')
parser.add_argument('-o', '--override', default=False, action='store_true', help='')
parser.add_argument('--no_tools', default=False, action='store_true', help='')
parser.add_argument('--full_obs', default=False, action='store_true',
                    help='Use Full-Obs ablation entry (task_allo scenario only)')
args = parser.parse_args()

# Example usage:
# python delib_collab/run_experiments.py --scenario cook --exp_levels 1 2 -g test_games -s 0 -e 60 --max_round 6 --max_character 3000 -m gpt-4o-mini --exp_name test -n 12 --n_procs_per_model 2
# python delib_collab/run_experiments.py --scenario cook --exp_levels 1 2 -g test_games -s 0 -e 60 -m gpt-4o-mini gpt-4.1-mini --exp_name test -n 12
# python delib_collab/run_experiments.py --scenario task_allo --exp_levels 1 -g task_allocation_demo_01 -s 0 -e 10 -m gpt-4o-mini --exp_name test --n_procs_per_model 2


def get_entry_file_path(scenario, exp_level, oracle=False, full_obs=False):
    """Get the entry file path based on scenario, level, oracle and full_obs flags."""
    if oracle:
        # Oracle agent mode (baseline)
        if scenario == 'cook':
            return os.path.join(agent_dir, 'cooking', 'oracle', 'entry.py')
        elif scenario == 'task_allo':
            # For task_allo, use baseline entry file
            if str(exp_level) == '1':
                return os.path.join(agent_dir, 'task_allocation', 'level1', 'baseline_entry.py')
            else:
                raise ValueError(f"Unsupported exp_level {exp_level} for task_allo oracle scenario. Only level 1 is supported.")
        else:
            raise ValueError(f"Unsupported scenario {scenario} for oracle mode")
    else:
        # Normal agent mode
        if scenario == 'cook':
            if str(exp_level) == '1':
                return os.path.join(agent_dir, 'cooking', 'level1', 'entry.py')
            elif str(exp_level) == '2':
                return os.path.join(agent_dir, 'cooking', 'level2', 'entry.py')
            else:
                raise ValueError(f"Unsupported exp_level {exp_level} for cook scenario")
        elif scenario == 'task_allo':
            if str(exp_level) == '1':
                if full_obs:
                    return os.path.join(agent_dir, 'task_allocation', 'level1', 'full_obs_entry.py')
                else:
                    return os.path.join(agent_dir, 'task_allocation', 'level1', 'entry.py')
            else:
                raise ValueError(f"Unsupported exp_level {exp_level} for task_allo scenario. Only level 1 is supported.")
        else:
            raise ValueError(f"Unsupported scenario {scenario}")


def run_single_game(kwargs):
    """Worker function to run a single game."""
    python_entry_file_path = kwargs["python_entry_file_path"]
    game_folder = kwargs["game_folder"]
    game_id = kwargs["game_id"]
    max_round = kwargs["max_round"]
    max_character = kwargs["max_character"]
    model = kwargs["model"]
    exp_name = kwargs["exp_name"]
    override = kwargs["override"]
    no_tools = kwargs["no_tools"]
    semaphores = kwargs["semaphores"]
    scenario = kwargs.get("scenario", "cook")
    exp_level = kwargs.get("exp_level", "1")
    oracle = kwargs.get("oracle", False)
    full_obs = kwargs.get("full_obs", False)
    
    # Acquire semaphore for this model
    semaphore = semaphores.get(model)
    if semaphore is None:
        print(f"Warning: No semaphore found for model {model}, proceeding without limit")
        semaphore = None
    
    try:
        if semaphore:
            semaphore.acquire()
        
        python_cmd = sys.executable
        # Build command for single game
        if oracle:
            # Oracle command format
            if scenario == 'cook':
                # Cook oracle needs --level parameter
                cmd_full = (f'{python_cmd} {python_entry_file_path} -g {game_folder} -s {game_id} -e {game_id + 1} '
                        f'--max_round {max_round} --max_character {max_character} -m {model} --exp_name {exp_name} '
                        f'--level level_{exp_level} {" -o " if override else ""} {" --no_tools" if no_tools else ""}')
            elif scenario == 'task_allo':
                # Task_allo baseline doesn't need --level parameter
                cmd_full = (f'{python_cmd} {python_entry_file_path} -g {game_folder} -s {game_id} -e {game_id + 1} '
                            f'--max_character {max_character} -m {model} --exp_name {exp_name}'
                            f' {" -o " if override else ""} {" --no_tools" if no_tools else ""}')
            else:
                raise ValueError(f"Unsupported scenario {scenario} for oracle mode")
        else:
            # Regular cook/task_allo/full_obs command format
            cmd_full = (f'{python_cmd} {python_entry_file_path} -g {game_folder} -s {game_id} -e {game_id + 1} '
                        f'--max_round {max_round} --max_character {max_character} -m {model} --exp_name {exp_name}'
                        f' {" -o " if override else ""} {" --no_tools" if no_tools else ""}')
        
        proc_name = multiprocessing.current_process().name
        print(f"[{proc_name}] Running game {game_id} with model {model}: {cmd_full}")
        
        # Execute subprocess
        result = subprocess.run(cmd_full, shell=True, capture_output=False, cwd=root_dir)
        
        if result.returncode == 0:
            print(f"[{proc_name}] Game {game_id} with model {model} completed successfully")
            return {'success': True, 'game_id': game_id, 'model': model}
        else:
            print(f"[{proc_name}] Game {game_id} with model {model} failed with return code {result.returncode}")
            return {'success': False, 'game_id': game_id, 'model': model, 'returncode': result.returncode}
    
    except Exception as e:
        print(f"[{multiprocessing.current_process().name}] Error running game {game_id} with model {model}: {e}")
        return {'success': False, 'game_id': game_id, 'model': model, 'error': str(e)}
    
    finally:
        # Release semaphore
        if semaphore:
            semaphore.release()


def generate_tasks(scenario, exp_levels, models, game_folder, start, end, max_round, 
                   max_character, exp_name, override, no_tools, semaphores, oracle=False, full_obs=False):
    """Generate all task combinations (model, game_id, exp_level)."""
    tasks = []
    
    for exp_level in exp_levels:
        python_entry_file_path = get_entry_file_path(scenario, exp_level, oracle=oracle, full_obs=full_obs)
        
        for model in models:
            _exp_name = exp_name + f"_{model}"
            
            for game_id in range(start, end):
                kwargs = {
                    'python_entry_file_path': python_entry_file_path,
                    'game_folder': game_folder,
                    'game_id': game_id,
                    'max_round': max_round,
                    'max_character': max_character,
                    'model': model,
                    'exp_name': _exp_name,
                    'override': override,
                    'no_tools': no_tools,
                    'semaphores': semaphores,
                    'scenario': scenario,
                    'exp_level': exp_level,
                    'oracle': oracle,
                    'full_obs': full_obs
                }
                tasks.append(kwargs)
    
    return tasks


def run_all():
    """Main function to run all experiments with dynamic task allocation."""
    scenario = args.scenario
    oracle = args.oracle
    full_obs = args.full_obs
    
    exp_levels = args.exp_levels
    num_process = args.num_process
    n_procs_per_model = args.n_procs_per_model
    game_folder = args.game_folder
    start = args.start
    end = args.end
    max_round = args.max_round
    max_character = args.max_character
    models = args.models
    exp_name = args.exp_name
    override = args.override
    no_tools = args.no_tools
    
    # Add _oracle suffix to exp_name if oracle flag is set and not already present
    if oracle and not exp_name.endswith('_oracle'):
        exp_name = exp_name + '_oracle'
    # Add _full_obs suffix to exp_name if full_obs flag is set and not already present
    if full_obs and not exp_name.endswith('_full_obs'):
        exp_name = exp_name + '_full_obs'
    
    if start is None or end is None:
        raise ValueError("Both --start and --end must be specified")
    
    # Validate scenario and levels
    if full_obs and scenario != 'task_allo':
        raise ValueError("--full_obs is only supported for --scenario task_allo")
    if scenario == 'task_allo':
        # Oracle (baseline) is now supported for task_allo
        for level in exp_levels:
            if str(level) != '1':
                raise ValueError(f"task_allo scenario only supports level 1, got {level}")
    elif scenario == 'cook':
        if oracle:
            # Oracle supports levels 1, 2, 3 for cook scenario
            for level in exp_levels:
                if str(level) not in ['1', '2', '3']:
                    raise ValueError(f"oracle mode for cook scenario supports levels 1, 2, 3, got {level}")
    
    # Determine n_procs_per_model and total_pool_size based on provided arguments
    n_models = len(models)
    
    if num_process is not None and n_procs_per_model is not None:
        # Both specified: prioritize n_procs_per_model
        expected_total = n_models * n_procs_per_model
        if num_process != expected_total:
            print(f"Warning: num_process ({num_process}) != n_models × n_procs_per_model ({expected_total})")
            print(f"Prioritizing n_procs_per_model={n_procs_per_model}, using {expected_total} total processes")
            print(f"Note: {num_process - expected_total} processes will not be used")
        total_pool_size = expected_total  # Use calculated total, not num_process
        max_concurrent_models = n_models  # All models can run concurrently
        print(f"\nConfiguration:")
        print(f"  Total processes requested: {num_process}")
        print(f"  Processes per model: {n_procs_per_model} (prioritized)")
        print(f"  Number of models: {n_models}")
        print(f"  -> Actual total processes: {total_pool_size}")
        print(f"  -> All {n_models} models will run concurrently")
    elif num_process is not None:
        # Only num_process specified: calculate n_procs_per_model
        n_procs_per_model = num_process // n_models
        if num_process % n_models != 0:
            print(f"Warning: {num_process} is not divisible by {n_models}, using {n_procs_per_model} processes per model")
            print(f"Actual total processes will be {n_models * n_procs_per_model}")
        total_pool_size = n_models * n_procs_per_model
        max_concurrent_models = num_process // n_procs_per_model if n_procs_per_model > 0 else 0
        print(f"\nConfiguration:")
        print(f"  Total processes: {num_process} (specified)")
        print(f"  Number of models: {n_models}")
        print(f"  -> Calculated: {n_procs_per_model} processes per model")
        print(f"  -> Actual total: {total_pool_size} processes")
        print(f"  -> Can run up to {max_concurrent_models} models concurrently")
    elif n_procs_per_model is not None:
        # Only n_procs_per_model specified: calculate total
        total_pool_size = n_models * n_procs_per_model
        print(f"\nConfiguration:")
        print(f"  Processes per model: {n_procs_per_model} (specified)")
        print(f"  Number of models: {n_models}")
        print(f"  -> Total processes: {total_pool_size}")
        print(f"  -> All {n_models} models will run concurrently")
    else:
        # Neither specified: use default
        n_procs_per_model = 2
        total_pool_size = n_models * n_procs_per_model
        print(f"\nConfiguration:")
        print(f"  Using default: {n_procs_per_model} processes per model")
        print(f"  Number of models: {n_models}")
        print(f"  -> Total processes: {total_pool_size}")
        print(f"  -> All {n_models} models will run concurrently")
    
    if n_procs_per_model <= 0:
        raise ValueError(f"n_procs_per_model must be > 0, got {n_procs_per_model}")
    
    # Create semaphores for each model using Manager
    manager = Manager()
    semaphores = manager.dict()
    for model in models:
        semaphores[model] = manager.Semaphore(n_procs_per_model)
    
    print(f"\nSemaphore configuration:")
    for model in models:
        print(f"  - {model}: {n_procs_per_model} concurrent processes")
    
    # Generate all tasks
    tasks = generate_tasks(
        scenario=scenario,
        exp_levels=exp_levels,
        models=models,
        game_folder=game_folder,
        start=start,
        end=end,
        max_round=max_round,
        max_character=max_character,
        exp_name=exp_name,
        override=override,
        no_tools=no_tools,
        semaphores=semaphores,
        oracle=oracle,
        full_obs=full_obs
    )
    
    print(f"Generated {len(tasks)} tasks total")
    print(f"  - {len(exp_levels)} level(s): {exp_levels}")
    print(f"  - {len(models)} model(s): {models}")
    print(f"  - {end - start} games: [{start}, {end})")
    
    # Run tasks with dynamic allocation
    print(f"\nStarting execution with {total_pool_size} workers...")
    results = []
    
    with multiprocessing.Pool(processes=total_pool_size) as pool:
        results = pool.map(run_single_game, tasks)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Execution Summary")
    print(f"{'='*60}")
    successful = sum(1 for r in results if r.get('success', False))
    failed = len(results) - successful
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for r in results:
            if not r.get('success', False):
                print(f"  - Game {r.get('game_id')} with model {r.get('model')}: {r.get('error', 'returncode ' + str(r.get('returncode', 'unknown')))}")


if __name__ == '__main__':
    run_all()
