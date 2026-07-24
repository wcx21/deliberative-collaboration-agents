#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import time

from langchain_core.prompts import PromptTemplate

from delib_collab.common.llm import model
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from delib_collab.prompts.cooking.level1.observe_legacy import OBSERVE_PROMPT

from delib_collab.common.parsing import extract_dict_from_string, auto_parse_text


class BuiltOracleAgent:
    # Use tag-based prompts and parser
    def __init__(self, LLM_model_name="model-name", agent_type='react'):
        self.LLM_model_name = LLM_model_name
        self.model = model(self.LLM_model_name)
        self.tools = []
        self.agent_type = agent_type

    def creat_agent(self, response_format):
        if self.agent_type == "react":
            if self.LLM_model_name == "model-name":
                agent = create_react_agent(self.model, tools=self.tools, response_format=response_format)
            else:
                agent = create_react_agent(self.model, tools=self.tools)
        elif self.agent_type == "chat":
            if self.LLM_model_name == "model-name" :
                agent = self.model.with_structured_output(response_format)
            else:
                agent = self.model
        return agent

    def prompt_format(self,prompt_template, **kwargs):
        template_variables = PromptTemplate(input_variables=[], template=prompt_template).input_variables
        valid_kwargs = {key: kwargs[key] for key in template_variables if key in kwargs}
        formatted_prompt = PromptTemplate(input_variables=template_variables, template=prompt_template).format_prompt(
            **valid_kwargs)
        return formatted_prompt.text

    def get_llm_response(self, agent, prompt_format, logger, tags):
        max_retry = 5
        time_sleep = 3
        # for unstable cheap api
        for _ in range(max_retry):
            try:
                response = agent.invoke({"messages": prompt_format})
                break
            except Exception as e:
                print(f"Error occurred due to unstable API, retrying... {_}")
                pass
                time.sleep(time_sleep)

        # response_ = extract_dict_from_string(response["messages"][-1].content, logger)
        response_ = auto_parse_text(response["messages"][-1].content, tags)

        input_tokens = 0  # Token counting depends on the selected model adapter
        output_tokens = 0  # Token counting depends on the selected model adapter
        return response, response_, input_tokens, output_tokens

    def single_step_log(self, name, input, output, output_, input_token_count, output_token_count,logger):
        logger.info('{} is called'.format(name))
        logger.info('{} input is {}'.format(name, input))
        logger.info('{} output is {}'.format(name, output))
        logger.info('{} parsed output is {}'.format(name, output_))
        logger.debug("{} input_tokens: {}".format(name, input_token_count))
        logger.debug("{} output_tokens: {}".format(name, output_token_count))

    def creat_subtask_agent(self,name, response_format, prompt_template,logger, **kwargs):
        subagent = self.creat_agent(response_format)
        prompt_format = self.prompt_format(prompt_template, **kwargs)
        tags = kwargs.get('tags', [])
        response, response_, input_tokens, output_tokens = self.get_llm_response(subagent, prompt_format, logger, tags)
        # self.single_step_log(name, prompt_format, response_, input_tokens, output_tokens, logger)
        self.single_step_log(name, prompt_format, response["messages"][-1].content, response_, input_tokens, output_tokens, logger)
        return response, response_, input_tokens, output_tokens, prompt_format
