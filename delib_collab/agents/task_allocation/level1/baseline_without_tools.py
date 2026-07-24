#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Task allocation Level 1 baseline processing class (without tools)."""

import os
import sys
import time
import traceback
import pickle
import json
import numpy as np

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.agents.task_allocation.level1.agent import BuiltAgentTaskAllocation

from delib_collab.agents.task_allocation.level1.schemas import DecisionProcessFormat

from delib_collab.common.logging_utils import setup_logger

from delib_collab.agents.task_allocation.level1.metrics import (
    calculate_game_metrics, print_game_metrics
)


BASELINE_DECISION_PROMPT = '''You are agent_2 (Leader), participating in a task allocation scenario.

== BASELINE EXPERIMENT ==
This is a baseline experiment where you have access to ALL information from all agents.
You need to make a final decision directly without negotiation.

== SCENARIO ==
You and 2 partners (agent_0 and agent_1) need to allocate {num_tasks} tasks among 3 agents.
The goal is to maximize total team value while satisfying resource constraints.

The task names you MUST use as JSON keys: {task_names}

== COMPLETE INFORMATION FROM ALL AGENTS ==

**Agent 0 (Worker1) Efficiency values:**

<agent_0_efficiency_values>
{agent_0_efficiency}
</agent_0_efficiency_values>

**Agent 1 (Worker2) Efficiency values:**

<agent_1_efficiency_values>
{agent_1_efficiency}
</agent_1_efficiency_values>

**Agent 2 (Leader - You) Efficiency values:**

<agent_2_efficiency_values>
{agent_2_efficiency}
</agent_2_efficiency_values>

== TOTAL RESOURCES ==
**CRITICAL**: 
- Private resources stay SEPARATE per agent (do NOT add them together)
- Public resources MUST be ADDED: total_public = agent_0_obs + agent_1_obs + agent_2_obs

<total_private_resources>
Agent 0: {agent_0_private_resources}
Agent 1: {agent_1_private_resources}
Agent 2: {agent_2_private_resources}
</total_private_resources>

<total_public_resources>
{total_public_resources}
</total_public_resources>

== TASK REQUIREMENTS ==
<task_requirements>
{task_requirements}
</task_requirements>
(Note: Missing resource = 0 required)

== YOUR TASK ==
Based on the complete information above, make a final decision on task allocation.

**CRITICAL - Resource Constraint Verification:**
Before proposing, verify: (1) Each agent's private resource demands ≤ their capacity, (2) Total public resource demand ≤ TOTAL pooled public resources (sum of all agents' observations). If constraints violated, adjust allocation.

**IMPORTANT**:
- Each task should be assigned to exactly one agent (or left unassigned if impossible due to resource constraints)
- Verify that your proposal does NOT exceed available resources
- Maximize total team value (assign tasks to agents who are most efficient at them)

**CRITICAL - Output Format Requirements**:
1. You MUST use XML tags with angle brackets (e.g., <reasoning_process>...</reasoning_process>)
2. Inside the tags that require JSON, output JSON DIRECTLY without markdown code blocks
3. NO ```json or ``` markers around the JSON

Please respond in the following format:

<reasoning_process>
(Your overall reasoning about the allocation, including resource constraint verification)
</reasoning_process>

<decision>
{{
    "accept_decision": true,
    "explanation": "Your reasoning for this allocation...",
    "final_decision": {{
        "task_name_1": "agent_X",
        "task_name_2": "agent_Y",
        ...
    }}
}}
</decision>

**IMPORTANT**: You MUST include the "final_decision" field with your allocation proposal.
'''


class ProcessTaskAllocationLevel1BaselineNoTools:
    """Task allocation Level 1 baseline processing class (without tools)."""
    
    def __init__(self, game, LLM_model_name="model-name", max_character=3000,
                 log_folder=None, log_name="task_allocation_level_1_baseline_no_tools", game_id=0, 
                 record_folder_name=None, game_level='level_1', gen_max_retries=5):
        """Initialize the processing class."""
        self.game = game
        self.LLM_model_name = LLM_model_name
        self.builtAgents = BuiltAgentTaskAllocation(LLM_model_name=self.LLM_model_name, agent_type='react')
        self.max_character = max_character
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.logger = setup_logger(log_name + "_game{}".format(game_id), log_folder=log_folder)
        self.logger.info("Current game is: {} (BASELINE - No Tools)".format(game_id))
        self.logger_dialogue = setup_logger(log_name + "_game{}_{}".format(game_id, "dialogue"), log_folder=log_folder)
        self.logger_final_allocation = setup_logger(log_name + "_game{}_{}".format(game_id, "final_allocation"),
                                                    log_folder=log_folder)
        self.logger_token = setup_logger(log_name + "_game{}_{}".format(game_id, "token"), log_folder=log_folder)
        
        self.max_retries = gen_max_retries
        self.record_folder_name = record_folder_name if record_folder_name else log_folder
        self.game_level = game_level
        self.record = {
            "game_level": self.game_level,
            "game_id": game_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "LLM_model_name": self.LLM_model_name,
            "baseline_type": "no_tools",
            "total_input_token_count": 0,
            "total_output_token_count": 0
        }
        result_path = os.path.join(root_dir, "result")
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        self.absolute_result_path = os.path.join(result_path, record_folder_name) if record_folder_name else None
        if self.absolute_result_path:
            os.makedirs(self.absolute_result_path, exist_ok=True)
            self.result_file_path = os.path.join(self.absolute_result_path, "full_record.pkl")
            self.short_result_file_path = os.path.join(self.absolute_result_path, "short_result.pkl")
        else:
            self.result_file_path = None
            self.short_result_file_path = None
        if os.path.exists(self.result_file_path) if self.result_file_path else False:
            with open(self.result_file_path, 'rb') as pkl_file:
                self.existing_data = pickle.load(pkl_file)
        else:
            self.existing_data = []

    def _get_agent_observation(self, agent_name):
        """Get the observation for the specified agent."""
        if agent_name == 'agent_0':
            obs = self.game.agent_0_obs
        elif agent_name == 'agent_1':
            obs = self.game.agent_1_obs
        elif agent_name == 'agent_2':
            obs = self.game.agent_2_obs
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")
        private_resources = obs.get('private_resources', {})
        public_resources = obs.get('public_resources', {})
        agent_idx = int(agent_name.split('_')[1])
        efficiency_dict = {}
        for task_idx, task_name in enumerate(self.game.tasks):
            efficiency = float(self.game.value_matrix[agent_idx, task_idx])
            efficiency = round(efficiency, 3)
            if efficiency > 0:
                efficiency_dict[task_name] = efficiency
        
        return private_resources, public_resources, efficiency_dict

    def _format_task_requirements(self):
        """Format task requirements into a JSON string."""
        processed_requirements = {}
        
        for task_name, requirements in self.game.task_requirements.items():
            processed_task = {}
            if 'agent_0' in requirements:
                agent_0_res = requirements['agent_0']
                non_zero_agent = {k: v for k, v in agent_0_res.items() if v != 0}
                processed_task.update(non_zero_agent)
            
            processed_requirements[task_name] = processed_task
        
        return json.dumps(processed_requirements, ensure_ascii=False, indent=2)

    @property
    def process(self):
        """Baseline process: give Leader all information to make a direct decision."""
        self.logger.info("=" * 80)
        self.logger.info("BASELINE EXPERIMENT - No Tools Version")
        self.logger.info("=" * 80)
        agent_0_private, agent_0_public, agent_0_efficiency = self._get_agent_observation('agent_0')
        agent_1_private, agent_1_public, agent_1_efficiency = self._get_agent_observation('agent_1')
        agent_2_private, agent_2_public, agent_2_efficiency = self._get_agent_observation('agent_2')
        total_public_resources = {}
        all_public_obs = [agent_0_public, agent_1_public, agent_2_public]
        for public_obs in all_public_obs:
            for resource_type, resource_value in public_obs.items():
                total_public_resources[resource_type] = total_public_resources.get(resource_type, 0) + resource_value
        task_requirements = self._format_task_requirements()
        task_names = self.game.tasks
        task_names_str = ", ".join([f'"{name}"' for name in task_names])
        
        kwargs = {
            "agent_name": "agent_2",
            "num_tasks": len(self.game.tasks),
            "task_names": task_names_str,
            "agent_0_private_resources": json.dumps(agent_0_private, ensure_ascii=False),
            "agent_0_public_resources": json.dumps(agent_0_public, ensure_ascii=False),
            "agent_0_efficiency": json.dumps(agent_0_efficiency, ensure_ascii=False),
            "agent_1_private_resources": json.dumps(agent_1_private, ensure_ascii=False),
            "agent_1_public_resources": json.dumps(agent_1_public, ensure_ascii=False),
            "agent_1_efficiency": json.dumps(agent_1_efficiency, ensure_ascii=False),
            "agent_2_private_resources": json.dumps(agent_2_private, ensure_ascii=False),
            "agent_2_public_resources": json.dumps(agent_2_public, ensure_ascii=False),
            "agent_2_efficiency": json.dumps(agent_2_efficiency, ensure_ascii=False),
            "total_public_resources": json.dumps(total_public_resources, ensure_ascii=False),
            "task_requirements": task_requirements
        }
        final_proposal = {}
        final_decision_input_tokens = 0
        final_decision_output_tokens = 0
        final_decision_prompt_format = ''
        final_decision_response = None
        final_decision_response_ = None
        
        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.info("Calling Leader (agent_2) to make final decision...")
                
                decision_response, decision_response_, decision_input_tokens, decision_output_tokens, decision_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="baseline_decision_agent",
                    response_format=DecisionProcessFormat,
                    prompt_template=BASELINE_DECISION_PROMPT,
                    logger=self.logger,
                    token_logger=self.logger_token,
                    tags=['decision'],
                    **kwargs
                )
                decision_response_ = decision_response_['decision']
                
                if 'decision' in decision_response_ and isinstance(decision_response_['decision'], dict):
                    self.logger.warning("Detected double-nested 'decision' field, unwrapping...")
                    decision_response_ = decision_response_['decision']
                
                if 'final_decision' in decision_response_ and decision_response_['final_decision'] is not None:
                    final_proposal = decision_response_['final_decision']
                    final_decision_input_tokens = decision_input_tokens
                    final_decision_output_tokens = decision_output_tokens
                    final_decision_prompt_format = decision_prompt_format
                    final_decision_response = decision_response
                    final_decision_response_ = decision_response_
                    
                    self.logger.info(f"Leader made final decision: {final_proposal}")
                    break
                else:
                    self.logger.warning("Leader did not provide final_decision, retrying...")
                    retries += 1
                    if retries < self.max_retries:
                        time.sleep(1)
                    else:
                        self.logger.error("Max retries reached. Leader did not provide final_decision.")
                        final_proposal = {}
                        
            except Exception as e:
                retries += 1
                self.logger.error(f"Error occurred while executing baseline decision (attempt {retries}/{self.max_retries}): {e}")
                self.logger.error(traceback.format_exc())
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using empty allocation as fallback.")
                    final_proposal = {}
                    break
                time.sleep(1)
        self.record["baseline_decision"] = {
            "prompt": final_decision_prompt_format,
            "response": final_decision_response,
            "response_after_format": final_decision_response_,
            "input_tokens": final_decision_input_tokens,
            "output_tokens": final_decision_output_tokens,
            "final_decision": final_proposal
        }
        
        self.total_input_token_count = final_decision_input_tokens
        self.total_output_token_count = final_decision_output_tokens
        self.record["total_input_token_count"] = self.total_input_token_count
        self.record["total_output_token_count"] = self.total_output_token_count
        agent_score = self.game.evaluate_allocation(final_proposal)
        max_reward = self.game.max_reward
        if max_reward > 0:
            score_ratio = np.round(agent_score / max_reward, 3)
        else:
            score_ratio = 0.0
        metrics_dict = calculate_game_metrics(
            final_allocation=final_proposal,
            game=self.game,
            max_reward=max_reward
        )
        nr_percent = metrics_dict['nr']
        nar_percent = metrics_dict['nar']
        scores = (agent_score, max_reward, score_ratio, nr_percent, nar_percent)
        game_metrics = {
            "final_reward": metrics_dict['final_reward'],
            "max_reward": metrics_dict['max_reward'],
            "nr": metrics_dict['nr'],
            "nar": metrics_dict['nar'],
            "is_valid": metrics_dict['is_valid'],
            "submenu": metrics_dict.get('submenu', {}),
            "submenu_reward": metrics_dict.get('submenu_reward', 0.0),
            "original_allocation_size": metrics_dict.get('original_allocation_size', len(final_proposal)),
            "submenu_size": metrics_dict.get('submenu_size', 0)
        }
        self.logger_final_allocation.info(f"Final proposal: {final_proposal}")
        self.logger_final_allocation.info(f"Best allocation (ground truth): {self.game.best_allocation_dict}")
        self.logger_final_allocation.info(f"Scores: {scores}")
        self.logger_final_allocation.info(f"Evaluation Metrics:")
        self.logger_final_allocation.info(f"  NR (Normalized Reward): {game_metrics['nr']:.2f}%")
        self.logger_final_allocation.info(f"  NAR (Normalized Adjusted Reward): {game_metrics['nar']:.2f}%")
        self.logger_final_allocation.info(f"  Is Valid: {game_metrics['is_valid']}")
        if not game_metrics['is_valid']:
            self.logger_final_allocation.info(f"  Submenu: {game_metrics['submenu']}")
            self.logger_final_allocation.info(f"  Submenu Reward: {game_metrics['submenu_reward']:.2f}")
        self.logger.info(f"Final proposal: {final_proposal}")
        self.logger.info(f"Scores: {scores}")
        self.record["scores"] = scores
        self.record["final_proposal"] = final_proposal
        self.record["final_score"] = agent_score
        self.record["round_count"] = 1
        self.record["evaluation_metrics"] = game_metrics
        if self.result_file_path:
            self.existing_data.append(self.record)
            with open(self.result_file_path, 'wb') as pkl_file:
                pickle.dump(self.existing_data, pkl_file)
            self.logger.info(f"Full record saved to: {self.result_file_path}")
        if self.short_result_file_path:
            short_result = {
                'final_proposal': final_proposal,
                'scores': scores,
                'rounds': 1,
                'evaluation_metrics': game_metrics
            }
            with open(self.short_result_file_path, 'wb') as f:
                pickle.dump(short_result, f)
            self.logger.info(f"Short result saved to: {self.short_result_file_path}")
        
        return final_proposal, agent_score, game_metrics

