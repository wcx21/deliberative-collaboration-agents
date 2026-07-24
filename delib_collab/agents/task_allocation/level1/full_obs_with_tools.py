#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Task Allocation Level 1 Full-Obs processor (with-tools variant).

Extends the no-tools Full-Obs class by exposing solver/calculator results to the LLM
via kwargs and using with-tools-specific prompt templates.
"""

import os
import sys
import json
import traceback

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)
sys.path.append(root_dir)

from delib_collab.agents.task_allocation.level1.full_obs_without_tools import (
    ProcessTaskAllocationLevel1FullObsNoTools
)

from delib_collab.prompts.task_allocation.level1.with_tools import (
    THINK_PROMPT_FULL_OBS, DECISION_PROMPT_FULL_OBS, TALK_PROMPT_FULL_OBS
)

from delib_collab.agents.task_allocation.level1.schemas import (
    ThinkingFormat, DecisionProcessFormat, MessageFormat
)

from delib_collab.agents.task_allocation import tools as tools_pool


class ProcessTaskAllocationLevel1FullObsWithTools(ProcessTaskAllocationLevel1FullObsNoTools):
    """
    Full-Obs Deliberation (with-tools variant).

    After OBSERVE bypass, solver/calculator results are written into kwargs
    (visible to the LLM agent) using with-tools Full-Obs prompt templates.
    Only overrides single_agent_step_process() from the no-tools parent class.
    """

    def __init__(self, game, LLM_model_name="model-name", max_conversation_rounds=10,
                 max_character=3000, log_folder=None,
                 log_name="task_allocation_level_1_full_obs_with_tools", game_id=0,
                 record_folder_name=None, game_level='level_1', gen_max_retries=3):
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

    def single_agent_step_process(self, **kwargs):
        """
        Full single-agent round (Full-Obs with-tools variant).

        OBSERVE is bypassed (GT injected directly). Solver/calculator results
        are written into kwargs (visible to LLM). Uses with-tools Full-Obs prompts.
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
            "bypass_reason": "Full-Obs ablation: ground-truth injected without LLM call",
            "total_resources": kwargs['total_resources'],
            "overall_preferences": kwargs['overall_preferences'],
        }

        # ================ Solver & Calculator (results exposed to LLM via kwargs) ========================
        try:
            best_allocation_raw, reward = tools_pool.agent_call_solver(
                self.game, kwargs['total_resources'], kwargs['overall_preferences']
            )
            kwargs['expected_reward'] = reward
            available_allocations_raw, insufficient_allocation_info_raw = tools_pool.agent_call_task_calculator(
                self.game, kwargs['total_resources']
            )
            kwargs['best_allocation'] = json.dumps(best_allocation_raw, ensure_ascii=False)
            kwargs['available_allocations'] = json.dumps(available_allocations_raw, ensure_ascii=False, indent=2)
            kwargs['insufficient_allocation_info'] = json.dumps(insufficient_allocation_info_raw, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Solver/Calculator call failed: {e}")
            best_allocation_raw = {}
            kwargs['expected_reward'] = 0.0
            available_allocations_raw = {}
            insufficient_allocation_info_raw = {}
            kwargs['best_allocation'] = json.dumps({}, ensure_ascii=False)
            kwargs['available_allocations'] = json.dumps({}, ensure_ascii=False)
            kwargs['insufficient_allocation_info'] = json.dumps({}, ensure_ascii=False)

        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"]["solver_calculator"] = {
            "best_allocation": best_allocation_raw,
            "expected_reward": kwargs.get('expected_reward', 0.0),
            "available_allocations": available_allocations_raw,
            "insufficient_allocation_info": insufficient_allocation_info_raw,
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
                self.logger.error(f"Error in think_agent (Full-Obs with_tools) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using default fallback for think_agent.")
                    self.logger.error(traceback.format_exc())
                    kwargs['agent_proposal'] = {}
                    kwargs['agent_proposal_explanation'] = 'Unable to generate proposal due to system error.'
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
                self.logger.error(f"Error in decision_agent (Full-Obs with_tools) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Using default fallback for decision_agent.")
                    self.logger.error(traceback.format_exc())
                    kwargs['decision_explanation'] = 'Unable to make decision due to system error. Rejecting.'
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
                self.logger.error(f"Error in talk_agent (Full-Obs with_tools) (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Default fallback for talk_agent.")
                    self.logger.error(traceback.format_exc())
                    talk_response_message_content = 'I encountered a system error and cannot communicate properly.'
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


if __name__ == '__main__':
    print("ProcessTaskAllocationLevel1FullObsWithTools module loaded successfully.")
    print("This is the Full-Obs Deliberation ablation variant (with_tools version).")
    print("Use task_allocation_level_1_full_obs_entry.py to run the full pipeline.")
