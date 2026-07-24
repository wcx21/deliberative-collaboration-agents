#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Task Allocation Level 1 Full-Obs processor (no-tools variant).

Ablation study baseline: bypasses OBSERVE (injects ground-truth directly),
exposes full observation to each agent, and uses Full-Obs prompt templates.
Solver/calculator results are recorded but hidden from the LLM.
"""

import os
import sys
import json
import builtins
import traceback
import numpy as np

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.agents.task_allocation.level1.process_without_tools import (
    ProcessTaskAllocationLevel1V1
)

from delib_collab.prompts.task_allocation.level1.without_tools import (
    THINK_PROMPT_FULL_OBS, DECISION_PROMPT_FULL_OBS, TALK_PROMPT_FULL_OBS
)

from delib_collab.agents.task_allocation.level1.schemas import (
    ThinkingFormat, DecisionProcessFormat, MessageFormat
)

from delib_collab.agents.task_allocation import tools as tools_pool


class ProcessTaskAllocationLevel1FullObsNoTools(ProcessTaskAllocationLevel1V1):
    """
    Full-Obs Deliberation (no-tools variant).

    Ablation study: injects ground-truth complete information to isolate
    the independent contribution of multi-round dialogue/distributed verification.
    """

    def __init__(self, game, LLM_model_name="model-name", max_conversation_rounds=10,
                 max_character=3000, log_folder=None,
                 log_name="task_allocation_level_1_full_obs_no_tools", game_id=0,
                 record_folder_name=None, game_level='level_1', gen_max_retries=3):
        """Initialize with parent class defaults; only log_name differs."""
        super().__init__(
            game=game,
            LLM_model_name=LLM_model_name,
            max_conversation_rounds=max_conversation_rounds,
            max_character=max_character,
            log_folder=log_folder,
            log_name=log_name,
            game_id=game_id,
            record_folder_name=record_folder_name,
            game_level=game_level,
            gen_max_retries=gen_max_retries,
        )

    def _get_agent_observation_full_obs(self, agent_name):
        """
        Get Full-Obs observation (ground-truth complete information).

        Returns:
            current_private_resources: This agent's private resources.
            current_public_resources: Global public resources (ground-truth).
            efficiency_dict_full: Nested {agent_X: {task: efficiency}} global dict.
        """
        agent_idx = int(agent_name.split('_')[1])
        if agent_name == 'agent_0':
            obs = self.game.agent_0_obs
        elif agent_name == 'agent_1':
            obs = self.game.agent_1_obs
        elif agent_name == 'agent_2':
            obs = self.game.agent_2_obs
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")

        current_private_resources = obs.get('private_resources', {})

        current_public_resources = self.game.game_resource_state.get('public_resources', {})

        efficiency_dict_full = {}
        for i in range(3):
            agent_key = f'agent_{i}'
            efficiency_dict_full[agent_key] = {}
            for task_idx, task_name in enumerate(self.game.tasks):
                val = float(self.game.value_matrix[i, task_idx])
                efficiency_dict_full[agent_key][task_name] = round(val, 3)

        return current_private_resources, current_public_resources, efficiency_dict_full

    def _build_full_obs_total_resources(self):
        """Build ground-truth global resource view for OBSERVE bypass injection."""
        return {
            'agent_private_resources': self.game.game_resource_state['agent_private_resources'],
            'public_resources': self.game.game_resource_state.get('public_resources', {})
        }

    def _build_full_obs_overall_preferences(self):
        """Build ground-truth global efficiency preferences for OBSERVE bypass injection."""
        preferences = {}
        for i in range(3):
            agent_key = f'agent_{i}'
            preferences[agent_key] = {}
            for task_idx, task_name in enumerate(self.game.tasks):
                val = float(self.game.value_matrix[i, task_idx])
                preferences[agent_key][task_name] = round(val, 3)
        return preferences

    def single_agent_step_process(self, **kwargs):
        """
        Full single-agent round (Full-Obs no-tools variant).

        OBSERVE is bypassed (GT injected directly, tokens = 0).
        Uses Full-Obs prompt templates for THINK/DECISION/TALK.
        """
        observer_input_tokens = 0
        observer_output_tokens = 0
        think_input_tokens = decision_input_tokens = talk_input_tokens = 0
        think_output_tokens = decision_output_tokens = talk_output_tokens = 0
        retries = 0
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']] = {}

        final_decision = None

        # ================ OBSERVE bypass (inject Ground-Truth directly) ========================
        kwargs['total_resources'] = self._build_full_obs_total_resources()
        kwargs['overall_preferences'] = self._build_full_obs_overall_preferences()
        kwargs['partner_resources'] = {}
        kwargs['partner_preferences'] = {}

        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"] = {
            "prompt": "BYPASSED (Full-Obs mode): GT injected directly",
            "response": "N/A",
            "response_after_format": {
                "total_resources": kwargs['total_resources'],
                "overall_preferences": kwargs['overall_preferences'],
            },
            "input_tokens": 0,
            "output_tokens": 0,
            "bypass_reason": "Full-Obs ablation: ground-truth information injected without LLM call",
            "total_resources": kwargs['total_resources'],
            "overall_preferences": kwargs['overall_preferences'],
        }

        # ================ Solver & Calculator (record only, hidden from agent) ========================
        try:
            best_allocation, reward = tools_pool.agent_call_solver(
                self.game, kwargs['total_resources'], kwargs['overall_preferences']
            )
            available_allocations, insufficient_allocation_info = tools_pool.agent_call_task_calculator(
                self.game, kwargs['total_resources']
            )
        except Exception as e:
            self.logger.warning(f"Solver/Calculator call failed (hidden record): {e}")
            best_allocation, reward = {}, 0.0
            available_allocations, insufficient_allocation_info = {}, {}

        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["tool_results_hidden"] = {
            "best_allocation": best_allocation,
            "reward": reward,
            "available_allocations": available_allocations,
            "insufficient_allocation_info": insufficient_allocation_info
        }

        # ================ THINK ========================
        retries = 0
        while retries < self.max_retries:
            try:
                think_response, think_response_, think_input_tokens, think_output_tokens, think_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="think_agent",
                    response_format=ThinkingFormat,
                    prompt_template=THINK_PROMPT_FULL_OBS,
                    logger=self.logger,
                    token_logger=self.logger_token,
                    tags=['proposal'],
                    **kwargs
                )
                think_response_ = think_response_['proposal']
                kwargs['agent_proposal'] = think_response_['allocation_proposal']
                kwargs['agent_proposal_explanation'] = think_response_['explanation']

                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"] = {
                    "prompt": think_prompt_format,
                    "response": think_response,
                    "response_after_format": think_response_,
                    "input_tokens": think_input_tokens,
                    "output_tokens": think_output_tokens,
                    "allocation_proposal": kwargs['agent_proposal']
                }
                break
            except Exception as e:
                retries += 1
                self.logger.error(f"Error in think_agent (Full-Obs) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using default fallback for think_agent.")
                    self.logger.error(traceback.format_exc())
                    kwargs['agent_proposal'] = {}
                    kwargs['agent_proposal_explanation'] = 'Unable to generate proposal due to system error. Proposing empty allocation as fallback.'
                    break

        # ================ DECISION ========================
        retries = 0
        accept_decision = False
        while retries < self.max_retries:
            try:
                if kwargs['partner_proposal'] is None or len(kwargs['partner_proposal']) == 0:
                    decision_response = decision_response_ = {
                        'decision': {
                            'accept_decision': False,
                            'explanation': "My partners have not proposed an allocation yet.",
                            'final_decision': None
                        }
                    }
                    decision_input_tokens, decision_output_tokens = 0, 0
                    decision_prompt_format = ''
                else:
                    decision_response, decision_response_, decision_input_tokens, decision_output_tokens, decision_prompt_format = self.builtAgents.creat_subtask_agent(
                        name="decision_agent",
                        response_format=DecisionProcessFormat,
                        prompt_template=DECISION_PROMPT_FULL_OBS,
                        logger=self.logger,
                        token_logger=self.logger_token,
                        tags=['decision'],
                        **kwargs
                    )

                decision_response_ = decision_response_['decision']

                if 'decision' in decision_response_ and isinstance(decision_response_['decision'], dict):
                    self.logger.warning("Detected double-nested 'decision' field, unwrapping...")
                    decision_response_ = decision_response_['decision']

                kwargs['decision_explanation'] = decision_response_['explanation']
                accept_decision = decision_response_['accept_decision']

                if 'final_decision' in decision_response_ and decision_response_['final_decision'] is not None:
                    final_decision = decision_response_['final_decision']

                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"] = {
                    "prompt": decision_prompt_format,
                    "response": decision_response,
                    "response_after_format": decision_response_,
                    "input_tokens": decision_input_tokens,
                    "output_tokens": decision_output_tokens,
                    "decision_explanation": decision_response_['explanation'],
                    "accept_decision": decision_response_['accept_decision'],
                    "final_decision": final_decision
                }
                break
            except Exception as e:
                retries += 1
                self.logger.error(f"Error in decision_agent (Full-Obs) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using default fallback for decision_agent.")
                    self.logger.error(traceback.format_exc())
                    kwargs['decision_explanation'] = 'Unable to make decision due to system error. Rejecting by default.'
                    accept_decision = False
                    break

        # ================ TALK ========================
        retries = 0
        talk_response_message_content = ''
        talk_input_tokens, talk_output_tokens = 0, 0

        should_talk = (final_decision is None)
        kwargs['accept_decision'] = accept_decision

        while retries < self.max_retries and should_talk:
            try:
                talk_response, talk_response_, talk_input_tokens, talk_output_tokens, talk_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="talk_agent",
                    response_format=MessageFormat,
                    prompt_template=TALK_PROMPT_FULL_OBS,
                    logger=self.logger,
                    token_logger=self.logger_token,
                    tags=['message_content'],
                    **kwargs
                )
                talk_response_message_content = talk_response_["message_content"]
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"] = {
                    "prompt": talk_prompt_format,
                    "response": talk_response,
                    "response_after_format": talk_response_,
                    "input_tokens": talk_input_tokens,
                    "output_tokens": talk_output_tokens,
                    "message_content": talk_response_["message_content"]
                }
                break
            except Exception as e:
                retries += 1
                self.logger.error(f"Error in talk_agent (Full-Obs) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using default fallback message for talk_agent.")
                    self.logger.error(traceback.format_exc())
                    talk_response_message_content = 'I apologize, but I encountered a system error and cannot communicate properly at this moment.'
                    break

        # ================ Token statistics ========================
        input_tokens = observer_input_tokens + think_input_tokens + decision_input_tokens + talk_input_tokens
        output_tokens = observer_output_tokens + think_output_tokens + decision_output_tokens + talk_output_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["current_round_input_tokens"] = input_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["current_round_output_tokens"] = output_tokens
        self.total_input_token_count += input_tokens
        self.total_output_token_count += output_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_response_message_content"] = talk_response_message_content
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_explanation"] = kwargs['decision_explanation']
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["accept_decision"] = accept_decision

        return (talk_response_message_content, kwargs['agent_proposal'], kwargs['agent_proposal_explanation'],
                input_tokens, output_tokens, accept_decision, final_decision)

    @property
    def process(self):
        """
        Main multi-round deliberation loop (Full-Obs variant).

        Uses _get_agent_observation_full_obs instead of parent's _get_agent_observation,
        with nested global efficiency dict and global public resources.
        """
        from delib_collab.agents.task_allocation.level1.metrics import (
            calculate_game_metrics, print_game_metrics
        )

        chat_history = ""
        partner_proposal = {}
        partner_explanation = ''
        final_proposal = {}

        agent_names = ['agent_0', 'agent_1', 'agent_2']

        for round in range(1, self.max_conversation_rounds + 1):
            self.logger.info("=======================round {}=================================".format(round))
            self.record["round {}".format(round)] = {}

            agent_0_private, agent_0_public, agent_0_eff_full = self._get_agent_observation_full_obs('agent_0')
            agent_1_private, agent_1_public, agent_1_eff_full = self._get_agent_observation_full_obs('agent_1')
            agent_2_private, agent_2_public, agent_2_eff_full = self._get_agent_observation_full_obs('agent_2')

            self.record["round {}".format(round)]['agent_0_obs'] = {
                'private_resources': agent_0_private,
                'public_resources': agent_0_public,
                'efficiency': agent_0_eff_full
            }
            self.record["round {}".format(round)]['agent_1_obs'] = {
                'private_resources': agent_1_private,
                'public_resources': agent_1_public,
                'efficiency': agent_1_eff_full
            }
            self.record["round {}".format(round)]['agent_2_obs'] = {
                'private_resources': agent_2_private,
                'public_resources': agent_2_public,
                'efficiency': agent_2_eff_full
            }

            task_requirements_str = self._format_task_requirements()

            # ================ Baseline-aligned info (same for all agents) ========================
            eff_flat = {}
            for i in range(3):
                agent_key = f'agent_{i}'
                eff_flat[agent_key] = {}
                for task_idx, task_name in enumerate(self.game.tasks):
                    val = float(self.game.value_matrix[i, task_idx])
                    val = builtins.round(val, 3)
                    if val > 0:
                        eff_flat[agent_key][task_name] = val

            all_private = self.game.game_resource_state['agent_private_resources']
            total_public = self.game.game_resource_state.get('public_resources', {})
            task_names_str = ", ".join([f'"{name}"' for name in self.game.tasks])

            baseline_aligned_kwargs = {
                "agent_0_efficiency": json.dumps(eff_flat['agent_0'], ensure_ascii=False),
                "agent_1_efficiency": json.dumps(eff_flat['agent_1'], ensure_ascii=False),
                "agent_2_efficiency": json.dumps(eff_flat['agent_2'], ensure_ascii=False),
                "agent_0_private_resources": json.dumps(all_private['agent_0'], ensure_ascii=False),
                "agent_1_private_resources": json.dumps(all_private['agent_1'], ensure_ascii=False),
                "agent_2_private_resources": json.dumps(all_private['agent_2'], ensure_ascii=False),
                "total_public_resources": json.dumps(total_public, ensure_ascii=False),
                "task_names": task_names_str,
            }

            # ================ Agent 0 (Worker1) ========================
            self.logger.info("---------------------agent_0 (Worker1)------------------------")
            agent_0_current_args = {
                **baseline_aligned_kwargs,
                "agent_name": agent_names[0],
                "round": round,
                "max_round": self.max_conversation_rounds,
                "max_character": self.max_character,
                "current_private_resources": json.dumps(agent_0_private, ensure_ascii=False),
                "current_public_resources": json.dumps(agent_0_public, ensure_ascii=False),
                "efficiency_dict": json.dumps(agent_0_eff_full, ensure_ascii=False),
                "chat_history": chat_history,
                "round_count": round,
                "task_requirements": task_requirements_str,
                "partner_proposal": partner_proposal,
                "partner_explanation": partner_explanation,
                "num_tasks": len(self.game.tasks)
            }
            (agent_0_chat, agent_0_proposal, agent_0_explanation,
             agent_0_input_tokens, agent_0_output_tokens,
             agent_0_accept_decision, agent_0_final_decision) = self.single_agent_step_process(**agent_0_current_args)

            formatted_chat = f"Round {round} ({agent_names[0]}): {agent_0_chat}\n"
            chat_history += formatted_chat
            self.logger_dialogue.info(formatted_chat)

            partner_proposal = agent_0_proposal
            partner_explanation = agent_0_explanation

            # ================ Agent 1 (Worker2) ========================
            self.logger.info("---------------------agent_1 (Worker2)------------------------")
            agent_1_current_args = {
                **baseline_aligned_kwargs,
                "agent_name": agent_names[1],
                "round": round,
                "max_round": self.max_conversation_rounds,
                "max_character": self.max_character,
                "current_private_resources": json.dumps(agent_1_private, ensure_ascii=False),
                "current_public_resources": json.dumps(agent_1_public, ensure_ascii=False),
                "efficiency_dict": json.dumps(agent_1_eff_full, ensure_ascii=False),
                "chat_history": chat_history,
                "round_count": round,
                "task_requirements": task_requirements_str,
                "partner_proposal": partner_proposal,
                "partner_explanation": partner_explanation,
                "num_tasks": len(self.game.tasks)
            }
            (agent_1_chat, agent_1_proposal, agent_1_explanation,
             agent_1_input_tokens, agent_1_output_tokens,
             agent_1_accept_decision, agent_1_final_decision) = self.single_agent_step_process(**agent_1_current_args)

            formatted_chat = f"Round {round} ({agent_names[1]}): {agent_1_chat}\n"
            chat_history += formatted_chat
            self.logger_dialogue.info(formatted_chat)

            partner_proposal = agent_1_proposal
            partner_explanation = agent_1_explanation

            # ================ Agent 2 (Leader) ========================
            self.logger.info("---------------------agent_2 (Leader)------------------------")
            agent_2_current_args = {
                **baseline_aligned_kwargs,
                "agent_name": agent_names[2],
                "round": round,
                "max_round": self.max_conversation_rounds,
                "max_character": self.max_character,
                "current_private_resources": json.dumps(agent_2_private, ensure_ascii=False),
                "current_public_resources": json.dumps(agent_2_public, ensure_ascii=False),
                "efficiency_dict": json.dumps(agent_2_eff_full, ensure_ascii=False),
                "chat_history": chat_history,
                "round_count": round,
                "task_requirements": task_requirements_str,
                "partner_proposal": partner_proposal,
                "partner_explanation": partner_explanation,
                "num_tasks": len(self.game.tasks),
                "worker_0_accept": agent_0_accept_decision,
                "worker_1_accept": agent_1_accept_decision
            }
            (agent_2_chat, agent_2_proposal, agent_2_explanation,
             agent_2_input_tokens, agent_2_output_tokens,
             agent_2_accept_decision, agent_2_final_decision) = self.single_agent_step_process(**agent_2_current_args)

            formatted_chat = f"Round {round} ({agent_names[2]}): {agent_2_chat}\n"
            chat_history += formatted_chat
            self.logger_dialogue.info(formatted_chat)

            if agent_2_final_decision is not None:
                self.logger.info(f"Leader (agent_2) made final decision: {agent_2_final_decision}")
                final_proposal = agent_2_final_decision
                break

            partner_proposal = agent_2_proposal
            partner_explanation = agent_2_explanation

            self.logger.debug(f"Round {round} (agent_0) input tokens: {agent_0_input_tokens}")
            self.logger.debug(f"Round {round} (agent_0) output tokens: {agent_0_output_tokens}")
            self.logger.debug(f"Round {round} (agent_1) input tokens: {agent_1_input_tokens}")
            self.logger.debug(f"Round {round} (agent_1) output tokens: {agent_1_output_tokens}")
            self.logger.debug(f"Round {round} (agent_2) input tokens: {agent_2_input_tokens}")
            self.logger.debug(f"Round {round} (agent_2) output tokens: {agent_2_output_tokens}")
            total_round_tokens = (agent_0_input_tokens + agent_1_input_tokens + agent_2_input_tokens +
                                  agent_0_output_tokens + agent_1_output_tokens + agent_2_output_tokens)
            self.logger.debug(f"Round {round} total tokens: {total_round_tokens}")

        if len(final_proposal) == 0:
            self.logger.warning("Max rounds reached without Leader's final decision.")
            if agent_2_proposal and len(agent_2_proposal) > 0:
                final_proposal = agent_2_proposal
                self.logger.info("Using Leader's (agent_2) last proposal as final allocation.")
            elif agent_1_proposal and len(agent_1_proposal) > 0:
                final_proposal = agent_1_proposal
                self.logger.info("Using Worker1's (agent_1) last proposal as final allocation.")
            elif agent_0_proposal and len(agent_0_proposal) > 0:
                final_proposal = agent_0_proposal
                self.logger.info("Using Worker0's (agent_0) last proposal as final allocation.")
            else:
                self.logger.warning("No valid proposal found from any agent. Using empty allocation as fallback.")
                final_proposal = {}

        agent_score = self.game.evaluate_allocation(final_proposal)
        max_reward = self.game.max_reward

        if max_reward > 0:
            score_ratio = np.round(agent_score / max_reward, 3)
        else:
            score_ratio = 0.0

        self.logger_final_allocation.info(f"Final proposal: {final_proposal}")
        self.logger_final_allocation.info(f"Best allocation (ground truth): {self.game.best_allocation_dict}")

        game_metrics = calculate_game_metrics(
            final_allocation=final_proposal,
            game=self.game,
            max_reward=max_reward
        )

        nr_percent = game_metrics['nr']
        nar_percent = game_metrics['nar']
        scores = (agent_score, max_reward, score_ratio, nr_percent, nar_percent)

        self.logger_final_allocation.info(f"Scores: {scores}")
        self.logger.info(f"Final proposal: {final_proposal}")
        self.logger.info(f"Scores: {scores}")

        self.logger_final_allocation.info(f"Evaluation Metrics:")
        self.logger_final_allocation.info(f"  NR (Normalized Reward): {game_metrics['nr']:.2f}%")
        self.logger_final_allocation.info(f"  NAR (Normalized Adjusted Reward): {game_metrics['nar']:.2f}%")
        self.logger_final_allocation.info(f"  Is Valid: {game_metrics['is_valid']}")
        if not game_metrics['is_valid']:
            self.logger_final_allocation.info(f"  Submenu: {game_metrics['submenu']}")
            self.logger_final_allocation.info(f"  Submenu Reward: {game_metrics['submenu_reward']:.2f}")

        print_game_metrics(self.record["game_id"], game_metrics)

        self.record["scores"] = scores
        self.record["final_proposal"] = final_proposal
        self.record["final_score"] = agent_score
        self.record["chat_history"] = chat_history
        self.record["evaluation_metrics"] = game_metrics

        self.logger.debug(f"Task used {round} rounds, total input tokens: {self.total_input_token_count}")
        self.logger.debug(f"Task used {round} rounds, total output tokens: {self.total_output_token_count}")

        self.record["total_input_token_count"] = self.total_input_token_count
        self.record["total_output_token_count"] = self.total_output_token_count
        self.record["round_count"] = round

        import pickle
        self.existing_data.append(self.record)
        with open(self.result_file_path, 'wb') as pkl_file:
            pickle.dump(self.existing_data, pkl_file)

        short_result = {
            'final_proposal': final_proposal,
            'scores': scores,
            'rounds': round,
            'evaluation_metrics': game_metrics
        }
        with open(self.short_result_file_path, 'wb') as f:
            pickle.dump(short_result, f)

        return final_proposal, scores, game_metrics


if __name__ == '__main__':
    print("ProcessTaskAllocationLevel1FullObsNoTools module loaded successfully.")
    print("This is the Full-Obs Deliberation ablation variant.")
    print("Use task_allocation_level_1_full_obs_entry.py to run the full pipeline.")
