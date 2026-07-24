#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import time
import traceback

from delib_collab.data_generation.cooking import load_games as load_games_v2

from delib_collab.paths import PROJECT_ROOT
root_dir = str(PROJECT_ROOT)

import numpy as np
import pickle

from delib_collab.agents.cooking.level1.agent import BuiltAgentLevel1
from delib_collab.agents.cooking.level1.schemas import ObseverFormat, ThinkingFormat, DecisionProcessFormat, \
    MessageFormat
from delib_collab.prompts.cooking.level1.with_tools import OBSERVE_PROMPT, THINK_PROMPT, DECISION_PROMPT, TALK_PROMPT

from delib_collab.agents.cooking.tools import agent_call_solver
from delib_collab.agents.cooking import tools as tools_pool
from delib_collab.common.logging_utils import setup_logger


class ProcessLevel1WithToolV1:
    def __init__(self, game, LLM_model_name="model-name", max_conversation_rounds=10, max_character=3000,
                 log_folder=None, log_name="level_1_v0_record", game_id=0, record_folder_name=None,
                 game_level='level_1', gen_max_retries=3):
        self.game = game
        self.LLM_model_name = LLM_model_name
        self.builtAgents = BuiltAgentLevel1(LLM_model_name=self.LLM_model_name, agent_type='react')
        self.max_conversation_rounds = max_conversation_rounds
        self.max_character = max_character
        self.total_input_token_count = 0
        self.total_output_token_count = 0
        self.logger = setup_logger(log_name + "_game{}".format(game_id), log_folder=log_folder)
        self.logger.info("Current game is: {}".format(game_id))
        self.logger_dialogue = setup_logger(log_name + "_game{}_{}".format(game_id, "dialogue"), log_folder=log_folder)
        self.logger_final_menu = setup_logger(log_name + "_game{}_{}".format(game_id, "final_menu"),
                                              log_folder=log_folder)
        self.max_retries = gen_max_retries
        self.record_folder_name = record_folder_name if record_folder_name else log_folder
        self.game_level = game_level
        self.record = {
            "game_level": self.game_level,
            "game_id": game_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        }
        result_path = os.path.join(root_dir, "result")
        if not os.path.exists(result_path):
            os.makedirs(result_path, exist_ok=True)
        self.absolute_result_path = os.path.join(result_path, record_folder_name)
        os.makedirs(self.absolute_result_path, exist_ok=True)
        self.result_file_path = os.path.join(self.absolute_result_path, "full_record.pkl")
        self.short_result_file_path = os.path.join(self.absolute_result_path, "short_result.pkl")
        if os.path.exists(self.result_file_path):
            with open(self.result_file_path, 'rb') as pkl_file:
                self.existing_data = pickle.load(pkl_file)
        else:
            self.existing_data = []

    def single_agent_step_process(self, **kwargs):
        observer_input_tokens = think_input_tokens = decision_input_tokens = talk_input_tokens = 0
        observer_output_tokens = think_output_tokens = decision_output_tokens = talk_output_tokens = 0
        retries = 0
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']] = {}

        # ================ OBSERVE ========================
        while retries < self.max_retries:
            try:
                observer_response, observer_response_, observer_input_tokens, observer_output_tokens, observer_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="observer_agent",
                    response_format=ObseverFormat,
                    prompt_template=OBSERVE_PROMPT,
                    logger=self.logger,
                    tags=['partner_ingredients', 'partner_preference', 'total_ingredients', 'overall_preference'],
                    **kwargs)
                kwargs['partner_ingredients'] = observer_response_['partner_ingredients']
                kwargs['partner_preference'] = observer_response_['partner_preference']
                kwargs['total_ingredients'] = observer_response_['total_ingredients']
                kwargs['overall_preference'] = observer_response_['overall_preference']
                assert isinstance(observer_response_['partner_ingredients'], dict) and isinstance(observer_response_['partner_preference'], dict)
                assert isinstance(observer_response_['total_ingredients'], dict) and isinstance(observer_response_['overall_preference'], dict)

                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"] = {}
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "prompt"] = observer_prompt_format
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "response"] = observer_response
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "response_after_format"] = observer_response_
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "input_tokens"] = observer_input_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "output_tokens"] = observer_output_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "partner_ingredients"] = kwargs['partner_ingredients']
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "partner_preference"] = kwargs['partner_preference']
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "total_ingredients"] = kwargs['total_ingredients']
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["observer_agent"][
                    "overall_preference"] = kwargs['overall_preference']
                break
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"Error occurred while executing observer_agent (attempt {retries}/{self.max_retries})"
                )
                self.logger.error(f"raw response: {observer_response['messages'][-1].content}")

                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Unable to retrieve observer_response.")
                    self.logger.error(traceback.format_exc())
                    raise e

        kwargs['best_menu'], reward = agent_call_solver(self.game, kwargs['total_ingredients'],
                                                        kwargs['overall_preference'])
        suff_dishes, insuff_dish_info = tools_pool.agent_call_dish_calculator(self.game, kwargs['total_ingredients'])
        kwargs['available_dishes'] = suff_dishes
        kwargs['unavailable_dishes_info'] = insuff_dish_info

        # ================ THINK ========================
        retries = 0
        while retries < self.max_retries:
            try:
                think_response, think_response_, think_input_tokens, think_output_tokens, think_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="think_agent",
                    response_format=ThinkingFormat,
                    prompt_template=THINK_PROMPT,
                    logger=self.logger,
                    tags=['proposal'],
                    **kwargs)
                think_response_ = think_response_['proposal']
                kwargs['agent_proposal'] = think_response_['menu_proposal']
                kwargs['agent_proposal_explanation'] = think_response_['explanation']

                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"] = {}
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"][
                    "prompt"] = think_prompt_format
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"][
                    "response"] = think_response
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"][
                    "response_after_format"] = think_response_
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"][
                    "input_tokens"] = think_input_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"][
                    "output_tokens"] = think_output_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["think_agent"]["menu_proposal"] = \
                kwargs['agent_proposal']
                break
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"Error occurred while executing think_agent (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Unable to retrieve think_response.")
                    self.logger.error(traceback.format_exc())
                    kwargs['agent_proposal'] = []
                    kwargs['agent_proposal_explanation'] = ''

        # ================ DECISION ========================
        retries = 0
        accept_decision = False
        while retries < self.max_retries:
            try:
                if kwargs['partner_proposal'] is None or len(kwargs['partner_proposal']) == 0:
                    decision_response = decision_response_ = {
                        'decision': {
                            'accept_decision': False,
                            'explanation': "My partner has not proposed a menu yet."
                        }
                    }
                    decision_input_tokens, decision_output_tokens = 0, 0
                    decision_prompt_format = ''
                else:
                    decision_response, decision_response_, decision_input_tokens, decision_output_tokens, decision_prompt_format = self.builtAgents.creat_subtask_agent(
                        name="decision_agent",
                        response_format=DecisionProcessFormat,
                        prompt_template=DECISION_PROMPT,
                        logger=self.logger,
                        tags=['decision'],
                        **kwargs)
                decision_response_ = decision_response_['decision']
                kwargs['decision_explanation'] = decision_response_['explanation']

                accept_decision = decision_response_['accept_decision']
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"] = {}
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "prompt"] = decision_prompt_format
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "response"] = decision_response
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "response_after_format"] = decision_response_
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "input_tokens"] = decision_input_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "output_tokens"] = decision_output_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "decision_explanation"] = decision_response_['explanation']
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_agent"][
                    "accept_decision"] = decision_response_['accept_decision']
                break
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"Error occurred while executing decision_agent (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Unable to retrieve decision_response.")
                raise e

        # ================ TALK ========================
        retries = 0
        talk_response_message_content = ''
        talk_input_tokens, talk_output_tokens = 0, 0
        while retries < self.max_retries and not accept_decision:
            # if accept_decision is true, no need to talk
            try:
                talk_response, talk_response_, talk_input_tokens, talk_output_tokens, talk_prompt_format = self.builtAgents.creat_subtask_agent(
                    name="talk_agent",
                    response_format=MessageFormat,
                    prompt_template=TALK_PROMPT,
                    logger=self.logger,
                    tags=['message_content'],
                    **kwargs)

                talk_response_message_content = talk_response_["message_content"]
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"] = {}
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"][
                    "prompt"] = talk_prompt_format
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"][
                    "response"] = talk_response
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"][
                    "response_after_format"] = talk_response_
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"][
                    "input_tokens"] = talk_input_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"][
                    "output_tokens"] = talk_output_tokens
                self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["talk_agent"]["message_content"] = \
                talk_response_["message_content"]
                break
            except Exception as e:
                retries += 1
                self.logger.error(
                    f"Error occurred while executing talk_agent (attempt {retries}/{self.max_retries}): {e}")
                if retries == self.max_retries:
                    self.logger.error("Max retries reached. Unable to retrieve talk_response.")

        input_tokens = observer_input_tokens + think_input_tokens + decision_input_tokens + talk_input_tokens
        output_tokens = observer_output_tokens + think_output_tokens + decision_output_tokens + talk_output_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']][
            "current_round_input_tokens"] = input_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']][
            "current_round_output_tokens"] = output_tokens
        self.total_input_token_count += input_tokens
        self.total_output_token_count += output_tokens
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']][
            "talk_response_message_content"] = talk_response_message_content
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["decision_explanation"] = kwargs[
            'decision_explanation']
        self.record["round {}".format(kwargs["round"])][kwargs['agent_name']]["accept_decision"] = accept_decision
        return (talk_response_message_content, kwargs['agent_proposal'], kwargs['agent_proposal_explanation'],
                input_tokens, output_tokens, accept_decision)

    @property
    def process(self):
        chat_history = ""
        partner_proposal = []
        partner_explanation = ''
        final_proposal = []

        agent_names = ['Agent 1', 'Agent 2']
        for round in range(1, self.max_conversation_rounds + 1):
            # start from 1 to let LLM know the progress better

            Alice_obs = self.game.agent_0_obs
            Alice_score = self.game.nl_agent_0_values
            Bob_obs = self.game.agent_1_obs
            Bob_score = self.game.nl_agent_1_values
            self.logger.info("=======================round {}=================================".format(round))
            self.record["round {}".format(round)] = {}
            self.record["round {}".format(round)]['Alice_obs'] = Alice_obs
            self.record["round {}".format(round)]['Alice_score'] = Alice_score
            self.record["round {}".format(round)]['Bob_obs'] = Bob_obs
            self.record["round {}".format(round)]['Bob_score'] = Bob_score
            self.logger.info("---------------------Agent Alice------------------------")
            agent_Alice_current_args = {
                "agent_name": agent_names[0],
                "round": round,
                "max_round": self.max_conversation_rounds,
                "max_character": self.max_character,
                "current_obs": Alice_obs,
                "score_dict": Alice_score,
                "chat_history": chat_history,
                "guest_name": "Guest 1",
                "partner_guest_name": "Guest 2",
                "round_count": round,
                "recipes": self.game.nl_recipes,
                "partner_proposal": partner_proposal,
                "partner_explanation": partner_explanation
            }
            Alice_chat, Alice_proposal, Alice_explanation, Alice_input_tokens, Alice_output_tokens, Alice_accept_decision = self.single_agent_step_process(
                **agent_Alice_current_args)
            formated_chat = f"Round {round} ({agent_names[0]}):{Alice_chat} \n"
            chat_history += formated_chat
            self.logger_dialogue.info(formated_chat)

            partner_proposal = Alice_proposal
            partner_explanation = Alice_explanation
            if Alice_accept_decision:
                self.logger.info(f"{agent_names[0]} agree {agent_names[1]}'s proposal, the menu is {Bob_proposal}")
                final_proposal = Bob_proposal  # Alice accept Bob's proposal, so break the loop.
                break

            self.logger.info("---------------------Agent Bob------------------------")
            agent_Bob_current_args = {
                "agent_name": "Bob",
                "round": round,
                "max_round": self.max_conversation_rounds,
                "max_character": self.max_character,
                "current_obs": Bob_obs,
                "score_dict": Bob_score,
                "chat_history": chat_history,
                "guest_name": "Guest 2",
                "partner_guest_name": "Guest 1",
                "round_count": round,
                "recipes": self.game.nl_recipes,
                "partner_proposal": partner_proposal,
                "partner_explanation": partner_explanation
            }
            Bob_chat, Bob_proposal, Bob_explanation, Bob_input_tokens, Bob_output_tokens, Bob_accept_decision = self.single_agent_step_process(
                **agent_Bob_current_args)
            chat_history += "Round {} (Bob):{} ".format(round, Bob_chat)
            self.logger_dialogue.info("Round {} (Bob):{} ".format(round, Bob_chat))
            if Bob_accept_decision:
                self.logger.info(f"{agent_names[1]} agree {agent_names[0]}'s proposal, the menu is {Alice_proposal}")
                final_proposal = Alice_proposal
                break
            partner_proposal = Bob_proposal
            partner_explanation = Bob_explanation

            self.logger.debug("Round {} (Alice) input tokens:{} ".format(round, Alice_input_tokens))
            self.logger.debug("Round {} (Alice) output tokens:{}".format(round, Alice_output_tokens))
            self.logger.debug("Round {} (Bob) input tokens:{} ".format(round, Bob_input_tokens))
            self.logger.debug("Round {} (Bob) output tokens:{}".format(round, Bob_output_tokens))
            self.logger.debug("Round {} total input tokens:{} ".format(round, Alice_input_tokens + Bob_input_tokens))
            self.logger.debug("Round {} total output tokens:{}".format(round, Alice_output_tokens + Bob_output_tokens))

        agent_score = self.game.evaluate_menu(final_proposal)
        flexible_agent_score = self.game.evaluate_menu_loose(final_proposal)
        max_reward = self.game.max_reward
        scores = (agent_score, flexible_agent_score, self.game.max_reward,
                  np.round(agent_score / max_reward, 3), np.round(flexible_agent_score / max_reward, 3))

        self.logger_final_menu.info("Final proposal: {}".format(final_proposal))
        self.logger_final_menu.info("Possible best menu: {}".format([self.game.dishes[idx] for idx in self.game.best_menu]))
        self.logger_final_menu.info(scores)
        self.logger.info("Final proposal: {}".format(final_proposal))
        self.logger.info(scores)

        self.record["scores"] = (agent_score, self.game.max_reward, np.round(agent_score / self.game.max_reward, 3))
        self.record["final_proposal"] = final_proposal
        self.record["final_score"] = agent_score
        self.record["final_score_flexible"] = flexible_agent_score
        self.record["chat_history"] = chat_history

        self.logger.debug("Task use {} rounds, total input tokens:{} ".format(round + 1, self.total_input_token_count))
        self.logger.debug("Task use {} rounds,total output tokens:{}".format(round + 1, self.total_output_token_count))

        self.record["total_input_token_count"] = self.total_input_token_count
        self.record["total_output_token_count"] = self.total_output_token_count
        self.record["round_count"] = round + 1

        self.existing_data.append(self.record)
        with open(self.result_file_path, 'wb') as pkl_file:
            pickle.dump(self.existing_data, pkl_file)

        short_result = {
            'final_proposal': final_proposal,
            'scores': scores,
            'rounds': round + 1
        }
        with open(self.short_result_file_path, 'wb') as f:
            pickle.dump(short_result, f)

        return final_proposal, scores


if __name__ == '__main__':
    game_folder = 'test_games'
    max_round = 6
    model = 'model-name'
    exp_name = 'base_check'
    override = True
    max_character = 1500
    n_games = 1

    print(f"Loading games from {game_folder}, {n_games} games")
    games = load_games_v2.load_games(game_folder, n_games=n_games, level=1)

    game = games[0]

    process_level_1_v0 = ProcessLevel1WithToolV1(
                game, LLM_model_name=model, max_conversation_rounds=max_round, max_character=max_character,
                log_folder=os.path.join(exp_name, 'level_1_with_tools'), log_name='v1', game_id=0,
                record_folder_name=os.path.join(exp_name, 'level_1_no_tools', f'game_{0}'), game_level='level_1'
            )
    final_proposal, scores = process_level_1_v0.process

    print(scores)
