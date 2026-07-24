#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import json
import re

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def parse_text(text, tags=None):
    '''
    tags = ['ingredients']
    return {
     'ingredients': "XXX"
    }

    ingredients = json.load(parsed_text['ingredients'])

    '''
    if "<step_1>" in text or "<step_2>" in text:
        structure_type = "step"
    elif "<reasoning_process>" in text and "<menu_proposal>" in text and "<explanation>" in text:
        structure_type = "menu_proposal"
    elif "<reasoning_process>" in text and "<accept_decision>" in text and "<explanation>" in text:
        structure_type = "accept_decision"
    elif "<thinking_process>" in text and "<message_content>" in text:
        structure_type = "message_content"
    else:
        return {}

    result = {}

    if structure_type == "step":
        step_pattern = re.compile(r"<(step_\d+)>(.*?)</\1>", re.DOTALL)
        tag_pattern = re.compile(r"<(.*?)>(.*?)</\1>")

        for step_match in step_pattern.finditer(text):
            step_name = step_match.group(1)
            step_content = step_match.group(2)

            result[step_name] = {}
            for tag_match in tag_pattern.finditer(step_content):
                tag_name = tag_match.group(1)
                tag_value = tag_match.group(2)
                result[step_name][tag_name] = tag_value

    elif structure_type == "menu_proposal":
        result = {
            "reasoning_process": re.search(r"<reasoning_process>(.*?)</reasoning_process>", text, re.DOTALL).group(1),
            "menu_proposal": re.search(r"<menu_proposal>(.*?)</menu_proposal>", text, re.DOTALL).group(1),
            "explanation": re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL).group(1),
        }

    elif structure_type == "accept_decision":
        result = {
            "reasoning_process": re.search(r"<reasoning_process>(.*?)</reasoning_process>", text, re.DOTALL).group(1),
            "accept_decision": re.search(r"<accept_decision>(.*?)</accept_decision>", text, re.DOTALL).group(1),
            "explanation": re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL).group(1),
        }

    elif structure_type == "message_content":
        result = {
            "thinking_process": re.search(r"<thinking_process>(.*?)</thinking_process>", text, re.DOTALL).group(1),
            "message_content": re.search(r"<message_content>(.*?)</message_content>", text, re.DOTALL).group(1),
        }

    return result


def auto_parse_text(text, tags: list, keep_tag=False):
    """
    Extract XML tag contents from text and parse as JSON.
    Handles common formatting issues like markdown code blocks and whitespace.
    """
    result_dict = dict()

    for tag in tags:
        pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
        result = pattern.search(text)
        result_text = result.group() if result else ""
        if not keep_tag:
            result_text = result_text.strip("<{}>".format(tag)).strip("</{}>".format(tag))
        result_text = result_text.strip()

        try:
            _result_text = json.loads(result_text)
        except:
            try:
                cleaned_text = result_text
                cleaned_text = re.sub(r'^```(?:json|python)?\s*\n?', '', cleaned_text, flags=re.IGNORECASE)
                cleaned_text = re.sub(r'\n?```\s*$', '', cleaned_text)
                cleaned_text = cleaned_text.strip()
                cleaned_text = cleaned_text.replace("'", '"')
                _result_text = json.loads(cleaned_text)
            except:
                try:
                    match = re.search(r'(\{.*\}|\[.*\])', cleaned_text, re.DOTALL)
                    if match:
                        extracted_text = match.group(1)
                        _result_text = json.loads(extracted_text)
                    else:
                        _result_text = result_text
                except:
                    _result_text = result_text
        
        result_dict[tag] = _result_text

    return result_dict


def extract_dict_from_string(input_string, logger):
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', input_string, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.error("Can not decode json, string is {}".format(input_string))
            return None
    else:
        logger.error("No valid dictionary content found, string is {}".format(input_string))
        return None

import time

def retry_operation(func, max_retries=3, delay=1, *args, **kwargs):
    """
    Retry a function call up to max_retries times with a delay between attempts.
    :param func: Function to execute
    :param max_retries: Maximum number of retries
    :param delay: Delay in seconds between retries
    :param args: Positional arguments for the target function
    :param kwargs: Keyword arguments for the target function
    :return: Return value on success
    """
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise
            time.sleep(delay)
            print(f"Retrying... ({retries}/{max_retries}) due to error: {e}")

def test_parse():
    raw_text = '''
This is a test text.

<total ingredients> 
{
    "a": "b"
}
</total ingredients> 

<overall preference> 
{
    "c": "d"
}
</overall preference> 

<message> 
Hello Happy World!
From Kokoro Tsurumaki
</message> 

'''
    result = auto_parse_text(raw_text, ["total ingredients", "overall preference", "message"])
    print(result)
    ingre = json.loads(result["total ingredients"])
    pref = json.loads(result["overall preference"])
    msg = result["message"]
    print(ingre, pref, msg)


if __name__ == '__main__':
    test_parse()
