#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Task allocation agent builder class."""

import time
import traceback

from langchain_core.prompts import PromptTemplate

from delib_collab.common.llm import model
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


from delib_collab.common.parsing import extract_dict_from_string, auto_parse_text

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, token counting will be skipped")


class BuiltAgentTaskAllocation:
    """Agent builder for the task allocation problem."""
    
    # Use tag-based prompts and parser
    def __init__(self, LLM_model_name="model-name", agent_type='react'):
        """Initialize the agent builder with an LLM model and agent type."""
        self.LLM_model_name = LLM_model_name
        self.model = model(self.LLM_model_name)
        self.tools = []
        self.agent_type = agent_type
        
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(LLM_model_name)
            except Exception as e:
                print(f"Warning: Failed to initialize tokenizer for the selected model: {e}")

    def creat_agent(self, response_format):
        """Create an agent object (react or chat mode)."""
        if self.agent_type == "react":
            if self.LLM_model_name == "model-name":
                
                #agent = create_react_agent(self.model, tools=self.tools,response_format=response_format)
                agent = create_react_agent(self.model, tools=self.tools)

            else:
                agent = create_react_agent(self.model, tools=self.tools)
        elif self.agent_type == "chat":
            if self.LLM_model_name == "model-name":
                
                #agent = self.model
                #agent = self.model.with_structured_output(response_format)
                agent = self.model
            else:
                agent = self.model
        return agent

    def prompt_format(self, prompt_template, **kwargs):
        """Format a prompt template with the given keyword arguments."""
        template_variables = PromptTemplate(input_variables=[], template=prompt_template).input_variables
        valid_kwargs = {key: kwargs[key] for key in template_variables if key in kwargs}
        formatted_prompt = PromptTemplate(input_variables=template_variables, template=prompt_template).format_prompt(
            **valid_kwargs)
        return formatted_prompt.text

    def get_llm_response(self, agent, prompt_format, logger, tags):
        """Get an LLM response, with retries on failure."""
        max_retry = 5
        time_sleep = 5
        # for unstable cheap api
        for _ in range(max_retry):
            try:
                response = agent.invoke({"messages": prompt_format})
                response_ = auto_parse_text(response["messages"][-1].content, tags)
                break
            except Exception as e:
                print(f"Error occurred due to unstable API, retrying... {_}")
                traceback.print_exc()
                logger.warning(traceback.format_exc())
                pass
                time.sleep(time_sleep)

        # response_ = extract_dict_from_string(response["messages"][-1].content, logger)
        response_ = auto_parse_text(response["messages"][-1].content, tags)

        input_tokens = 0  # Token counting depends on the selected model adapter
        output_tokens = 0  # Token counting depends on the selected model adapter
        return response, response_, input_tokens, output_tokens

    def _validate_response(self, name, response_, tags, logger=None):
        """Validate whether the parsed response contains the required fields."""
        if not isinstance(response_, dict):
            if logger:
                logger.warning(f"{name} validation failed: response_ is not a dict, type is {type(response_)}")
            return False
        
        if name == "observer_agent":
            required_fields = ['partner_resources', 'partner_preferences', 'total_resources', 'overall_preferences']
            for field in required_fields:
                if field not in response_:
                    if logger:
                        logger.warning(f"{name} validation failed: missing field '{field}'. Available keys: {list(response_.keys())}")
                    return False
                if not response_[field]:
                    if logger:
                        logger.warning(f"{name} validation failed: field '{field}' is empty. Value: {response_[field]}")
                    return False
        elif name == "think_agent":
            if 'proposal' not in response_:
                if logger:
                    logger.warning(f"{name} validation failed: missing 'proposal'. Available keys: {list(response_.keys())}")
                return False
            proposal = response_['proposal']
            if not isinstance(proposal, dict) or 'allocation_proposal' not in proposal:
                if logger:
                    logger.warning(f"{name} validation failed: 'proposal' is not a valid dict or missing 'allocation_proposal'. Proposal: {proposal}")
                return False
            
            # if not proposal.get('allocation_proposal'):
            #     return False
        elif name == "decision_agent":
            if 'decision' not in response_:
                if logger:
                    logger.warning(f"{name} validation failed: missing 'decision'. Available keys: {list(response_.keys())}")
                return False
            decision = response_['decision']
            if not isinstance(decision, dict) or 'accept_decision' not in decision:
                if logger:
                    logger.warning(f"{name} validation failed: 'decision' is not a valid dict or missing 'accept_decision'. Decision: {decision}")
                return False
        elif name == "talk_agent":
            if 'message_content' not in response_:
                if logger:
                    logger.warning(f"{name} validation failed: missing 'message_content'. Available keys: {list(response_.keys())}")
                return False
            if not response_['message_content']:
                if logger:
                    logger.warning(f"{name} validation failed: 'message_content' is empty.")
                return False
        
        return True

    def _tokenize_text(self, text):
        """Tokenize text and return token count."""
        if self.tokenizer is None:
            return 0, []
        try:
            if isinstance(text, list):
                text = "\n".join([str(msg) for msg in text])
            tokens = self.tokenizer.encode(str(text), add_special_tokens=True)
            return len(tokens), tokens
        except Exception as e:
            print(f"Warning: Tokenization failed: {e}")
            return 0, []

    def single_step_log(self, name, input, output, output_, input_token_count, output_token_count, logger):
        """Log a single agent step (input, output, token counts)."""
        logger.info('{} is called'.format(name))
        logger.info('{} input is {}'.format(name, input))
        logger.info('{} output is {}'.format(name, output))
        logger.info('{} parsed output is {}'.format(name, output_))
        logger.debug("{} input_tokens: {}".format(name, input_token_count))
        logger.debug("{} output_tokens: {}".format(name, output_token_count))

    def _get_default_response(self, name, **kwargs):
        """Return a default fallback response for the given agent name."""
        if name == "observer_agent":
            num_agents = 3
            agent_names = [f'agent_{i}' for i in range(num_agents)]
            return {
                'partner_resources': {agent: {} for agent in agent_names if agent != kwargs.get('agent_name', 'agent_0')},
                'partner_preferences': {agent: {} for agent in agent_names if agent != kwargs.get('agent_name', 'agent_0')},
                'total_resources': {
                    'agent_private_resources': {agent: {} for agent in agent_names},
                    'public_resources': {}
                },
                'overall_preferences': {agent: {} for agent in agent_names}
            }
        elif name == "think_agent":
            return {
                'proposal': {
                    'allocation_proposal': {},
                    'explanation': 'Unable to generate proposal due to system error. Proposing empty allocation as fallback.'
                }
            }
        elif name == "decision_agent":
            return {
                'decision': {
                    'accept_decision': False,
                    'explanation': 'Unable to make decision due to system error. Rejecting by default.',
                    'final_decision': None
                }
            }
        elif name == "talk_agent":
            return {
                'message_content': 'I apologize, but I encountered a system error and cannot communicate properly at this moment.'
            }
        else:
            return {}

    def creat_subtask_agent(self, name, response_format, prompt_template, logger, token_logger=None, **kwargs):
        """Create and run a subtask agent (OBSERVE, THINK, DECISION, or TALK)."""
        prompt_format = self.prompt_format(prompt_template, **kwargs)
        tags = kwargs.get('tags', [])
        max_retries = 5
        
        tokenizer_input_token_count, tokenizer_input_tokens = self._tokenize_text(prompt_format)
        if token_logger:
            token_logger.info(f"=== {name} Input Tokenization ===")
            token_logger.info(f"Input text length: {len(prompt_format)} characters")
            token_logger.info(f"tokenizer input tokens: {tokenizer_input_token_count}")
            token_logger.debug(f"Input text (first 500 chars): {prompt_format[:500]}...")
        
        print(f"start to get LLM response for {name}...")
        
        subagent = self.creat_agent(response_format)
        total_input_tokens = 0
        total_output_tokens = 0
        
        for attempt in range(max_retries):
            try:
                response, response_, input_tokens, output_tokens = self.get_llm_response(subagent, prompt_format, logger, tags)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                output_content = response["messages"][-1].content if response and "messages" in response and len(response["messages"]) > 0 else ""
                tokenizer_output_token_count, tokenizer_output_tokens = self._tokenize_text(output_content)
                if token_logger:
                    token_logger.info(f"=== {name} Output Tokenization (attempt {attempt + 1}) ===")
                    token_logger.info(f"Output text length: {len(output_content)} characters")
                    token_logger.info(f"tokenizer output tokens: {tokenizer_output_token_count}")
                    token_logger.info(f"API input tokens: {input_tokens}")
                    token_logger.info(f"API output tokens: {output_tokens}")
                    token_logger.debug(f"Output text (first 500 chars): {output_content[:500]}...")
                
                if self._validate_response(name, response_, tags, logger):
                    print(f"finished getting LLM response for {name} (attempt {attempt + 1}).")
                    if token_logger:
                        token_logger.info(f"=== {name} Final Summary ===")
                        token_logger.info(f"Total API input tokens: {total_input_tokens}")
                        token_logger.info(f"Total API output tokens: {total_output_tokens}")
                        token_logger.info(f"tokenizer input tokens: {tokenizer_input_token_count}")
                        token_logger.info(f"tokenizer output tokens: {tokenizer_output_token_count}")
                        token_logger.info("=" * 50)
                    self.single_step_log(name, prompt_format, response["messages"][-1].content, response_, total_input_tokens, total_output_tokens, logger)
                    return response, response_, total_input_tokens, total_output_tokens, prompt_format
                else:
                    if attempt == 0:
                        logger.warning(f"{name} response validation failed on first attempt, retrying...")
                    else:
                        logger.warning(f"{name} response validation failed on attempt {attempt + 1}, retrying...")
                    
                    logger.debug(f"{name} raw LLM output (attempt {attempt + 1}): {response['messages'][-1].content[:500]}...")
                    logger.debug(f"{name} parsed response (attempt {attempt + 1}): {response_}")
                    
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error occurred while executing {name} (attempt {attempt + 1}/{max_retries}): {e}")
                logger.error(traceback.format_exc())
                if attempt == max_retries - 1:
                    logger.error(f"Max retries reached for {name}. Returning default fallback response.")
                    if token_logger:
                        token_logger.error(f"=== {name} Exception After All Retries ===")
                        token_logger.error(f"tokenizer input tokens: {tokenizer_input_token_count}")
                        token_logger.error(f"Exception: {e}")
                        token_logger.error("=" * 50)
                    response = {"messages": [{"content": ""}]}
                    response_ = self._get_default_response(name, **kwargs)
                    logger.warning(f"Using default fallback response for {name}: {response_}")
                    self.single_step_log(name, prompt_format, "", response_, total_input_tokens, total_output_tokens, logger)
                    return response, response_, total_input_tokens, total_output_tokens, prompt_format
                time.sleep(1)
        
        logger.error(f"All {max_retries} attempts failed for {name}. Returning default fallback response.")
        if token_logger:
            token_logger.error(f"=== {name} Failed After All Retries ===")
            token_logger.error(f"tokenizer input tokens: {tokenizer_input_token_count}")
            token_logger.error("=" * 50)

        response = {"messages": [{"content": ""}]}
        response_ = self._get_default_response(name, **kwargs)
        logger.warning(f"Using default fallback response for {name}: {response_}")
        self.single_step_log(name, prompt_format, "", response_, total_input_tokens, total_output_tokens, logger)
        return response, response_, total_input_tokens, total_output_tokens, prompt_format
