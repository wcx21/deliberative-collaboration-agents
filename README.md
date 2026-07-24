# Deliberative Collaboration

Code for studying **multi-agent deliberative collaboration**: LLM agents that
each hold only a *partial* view of the world and must reach a joint decision
through several rounds of natural-language negotiation.

The repository covers two tasks:

- **Cooking collaboration** — agents negotiate a shared menu under partial
  observations of available ingredients and of each guest's preferences.
- **Task allocation** — agents negotiate who does which task under private and
  public resource constraints.

The main Python package is `delib_collab`. It ships experiment runners, the
agents and prompts for each task, data-generation utilities, example datasets,
and evaluation scripts.

## Experiment modes

Both tasks share the same set of orthogonal conditions, selected with flags on
the runner:

| Mode | Flag | Description |
| --- | --- | --- |
| Deliberation | *(default)* | Partial-observation agents negotiate over several rounds. |
| With / without solver tools | `--no_tools` to disable | Whether agents may call a solver tool during deliberation. |
| Oracle baseline | `--oracle` | Centralized, full-observation baseline (no negotiation). |
| Full-observation ablation | `--full_obs` | Negotiation but with full observations (task allocation only). |


## Installation

```bash
git clone <repo-url>
cd <repo-name>

conda create -n delib-collab python=3.10 -y
conda activate delib-collab

pip install -r requirements.txt
export PYTHONPATH=$PWD
```

## Model configuration

All models are served through an **OpenAI-compatible endpoint**. Configure it
with two environment variables:

```bash
export API_KEY=...                          # credential for the endpoint (required)
export API_BASE_URL=https://your-gateway/v1 # optional; omit to use OpenAI's default
```

The adapter in [`delib_collab/common/llm.py`](delib_collab/common/llm.py) builds
a `ChatOpenAI` client against `API_BASE_URL` with `API_KEY`. Select the model at
runtime with `-m/--model`.

The model identifiers reported in the paper are resolved through
`MODEL_REGISTRY` in `llm.py`, which maps each id to its upstream model name and
the extra arguments needed to reproduce our setting (we run every model with its
internal *thinking* / reasoning disabled):

```text
gpt-5.1, gpt-4.1-mini, glm-4.7, deepseek-v3.2,
qwen3-next-80b, qwen3-32b, qwen3-30b
```

Any identifier not in the registry is passed through unchanged, so a model your
endpoint serves directly also works (e.g. `-m gpt-4o-mini` for a quick smoke
test).

Notes:

- The paper's experiments used several private OpenAI-compatible gateways; this
  release unifies them behind a single `API_BASE_URL`. Point it at any endpoint
  (an aggregating proxy, a self-hosted vLLM server, etc.) that serves the
  requested model names.
- If your gateway exposes a model under a different name, edit its entry in
  `MODEL_REGISTRY` (the `model` field) to match.

## Project layout

```text
.
|-- delib_collab/
|   |-- run_experiments.py          # main multi-process batch runner
|   |-- paths.py                    # centralized project/data paths
|   |-- common/                     # config, LLM adapter, logging, parsing helpers
|   |-- agents/
|   |   |-- cooking/                # cooking agents, per-level entries, tools
|   |   `-- task_allocation/        # task-allocation agents, entries, tools
|   |-- prompts/
|   |   |-- cooking/
|   |   `-- task_allocation/
|   `-- data_generation/
|       |-- cooking/
|       `-- task_allocation/
|-- data/
|   |-- cooking/                    # cooking source data and game pickles
|   `-- task_allocation/            # task-allocation source data and game pickles
`-- scripts/
    |-- evaluation/                 # result statistics and token accounting
```

## Running experiments

The single entry point is the batch runner:

```bash
python delib_collab/run_experiments.py [options]
```

Run it from the `open_code` directory with `PYTHONPATH=$PWD` set (see
[Installation](#installation)); each new shell needs it. 

The example datasets are bundled in the repo (`test_games` for cooking,
`task_allocation_games` and `task_allocation_demo_01` for task allocation, under
`data/<task>/games/`), so the commands below run as-is. See
[Data generation](#data-generation) only if you want to regenerate or extend them.

It expands a Cartesian product of `levels x models x game-range` into
independent single-game jobs and runs them across a process pool, with a
per-model concurrency limit.

Key arguments:

- `--scenario cook | task_allo`
- `--exp_levels 1 2` (cooking) or `--exp_levels 1` (task allocation)
- `-g / --game_folder` — dataset folder under `data/<task>/games/`
- `-s / -e` — game id range `[start, end)`
- `-m / --models` — one or more model identifiers (runs each)
- `--no_tools` — disable solver tool calls
- `--oracle` — run the centralized oracle baseline
- `--full_obs` — run the task-allocation full-observation ablation
- `--n_procs_per_model` — concurrent games per model (or `-n` for a total)
- `-o / --override` — overwrite existing results

### Cooking

Level 1, no tools, single game (quick smoke test):

```bash
python delib_collab/run_experiments.py \
  --scenario cook --exp_levels 1 \
  -g test_games -s 0 -e 1 \
  -m gpt-4o-mini --exp_name cook_l1_debug \
  --n_procs_per_model 1 --no_tools -o
```

Levels 1-2 with tools, 10 games:

```bash
python delib_collab/run_experiments.py \
  --scenario cook --exp_levels 1 2 \
  -g test_games -s 0 -e 10 \
  -m gpt-4o-mini --exp_name cook_tools \
  --n_procs_per_model 2 -o
```

Oracle baseline:

```bash
python delib_collab/run_experiments.py \
  --scenario cook --oracle --exp_levels 1 \
  -g test_games -s 0 -e 1 \
  -m gpt-4o-mini --exp_name cook_oracle \
  --n_procs_per_model 1 --no_tools -o
```

### Task allocation

Without tools:

```bash
python delib_collab/run_experiments.py \
  --scenario task_allo --exp_levels 1 \
  -g task_allocation_demo_01 -s 0 -e 1 \
  -m gpt-4o-mini --exp_name task_allo_debug \
  --n_procs_per_model 1 --no_tools -o
```

With tools (full dataset):

```bash
python delib_collab/run_experiments.py \
  --scenario task_allo --exp_levels 1 \
  -g task_allocation_games -s 0 -e 10 \
  -m gpt-4o-mini --exp_name task_allo_tools \
  --n_procs_per_model 2 -o
```

Oracle baseline / full-observation ablation: add `--oracle` or `--full_obs`
(with `--no_tools`) to the no-tools command above.

## Outputs

For each run, `--exp_name` is suffixed with the model id, so records and logs
land under:

```text
result/<exp_name>_<model>/<level_dir>/game_<id>/short_result.pkl
logs/<exp_name>_<model>/<level_dir>/...
```



## Data generation

Cooking games (source under `data/cooking/source/full_data/`):

```bash
python -m delib_collab.data_generation.cooking.generate_games
```

Task-allocation demo games:

```bash
python -m delib_collab.data_generation.task_allocation.generate_demo_games
```

Task-allocation full dataset (source under
`data/task_allocation/source/database/`):

```bash
python -m delib_collab.data_generation.task_allocation.generate_games
```

Generated pickles are written under `data/<task>/games/<output_dir>/games/`.
The default scripts sample many candidate games; reduce `n_seed` / `n_games`
inside the script (or import and call the generation functions directly) for
quick debugging.

## Evaluation and analysis

Once a run has produced result files under `result/<exp_name>_<model>/`, the
scripts in `scripts/evaluation/` aggregate them.

### Cooking

`stat_cooking.py` is driven by the configuration block at the bottom of the
file. Edit it to match your run — set `root_exp_name` to your `--exp_name`,
list the `models` you ran, and set `levels`, `game_folder`, and `n_games` — then
run it (optionally restrict the game range with `-s/-e`):

```bash
python scripts/evaluation/stat_cooking.py -s 0 -e 60
```

It reads `result/<exp_name>_<model>/level_<L>_<with|no>_tools/game_<id>/` for
every model and level, prints per-model metrics (Normalized Reward, NAR, VR,
state/value estimation accuracy, hallucination rate, ...), and writes a summary
to `result/<exp_name>_statistics.txt`. The second `run_statistics` call in the
same block aggregates the `--oracle` baseline; comment it out if you did not run
an oracle experiment.

### Task allocation

`stat_task_allocation.py` can evaluate a single result directory directly:

```bash
python scripts/evaluation/stat_task_allocation.py \
  result/<exp_name>_<model>/task_allocation_level_1_no_tools
```

To aggregate across several models at once, edit the `--run-full` block at the
bottom of the file (same `exp_name` / `models` pattern as cooking) and run with
`--run-full`.

## Citation

If you find this work useful, please cite our paper
([arXiv:2607.06157](https://arxiv.org/abs/2607.06157)):

```bibtex
@article{wang2026llm,
  title={LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability},
  author={Wang, Chenxu and Yang, Yongkun and Du, Boyuan and Lin, Shiwei and Liu, Huaping},
  journal={arXiv preprint arXiv:2607.06157},
  year={2026}
}
```

