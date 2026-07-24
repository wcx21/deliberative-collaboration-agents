#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""OpenAI-compatible chat model adapter.

Every model in the paper was served through an OpenAI-compatible endpoint, so
the adapter always builds a ``ChatOpenAI`` client pointed at ``API_BASE_URL``
with ``API_KEY``. ``MODEL_REGISTRY`` maps the model identifiers used throughout
the repo (and reported in the paper) to the upstream model name plus the extra
arguments needed to reproduce the paper's setting.

Reproducibility note: the experiments run every model with its internal
"thinking" / reasoning disabled, so comparisons are made on direct answers.
That is encoded per model below (``reasoning_effort="none"`` for the GPT-5.x
family, ``extra_body={"enable_thinking": False}`` for the open-weight models
that expose a thinking switch).

The paper used several private OpenAI-compatible gateways; this release unifies
them behind a single ``API_BASE_URL``. Point it at any endpoint that serves the
requested model names, and adjust ``MODEL_REGISTRY`` if your gateway exposes
them under different names.
"""

import json
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from delib_collab.common.config import API_BASE_URL, API_KEY


# id used in the repo / paper -> upstream model name + extra ChatOpenAI kwargs
MODEL_REGISTRY = {
    "gpt-5.1": {"model": "gpt-5.1", "reasoning_effort": "none"},
    "gpt-4.1-mini": {"model": "gpt-4.1-mini"},
    "glm-4.7": {"model": "GLM-4.7", "extra_body": {"enable_thinking": False}},
    "deepseek-v3.2": {
        "model": "DeepSeek-V3.2",
        "extra_body": {"enable_thinking": False},
    },
    "qwen3-next-80b": {"model": "Qwen3-Next-80B-A3B-Instruct"},
    "qwen3-32b": {"model": "Qwen3-32B", "extra_body": {"enable_thinking": False}},
    "qwen3-30b": {"model": "Qwen3-30B-A3B-Instruct-2507"},
}


def model(LLM_model_name, temperature=0.3):
    if not API_KEY:
        raise RuntimeError(
            "API_KEY environment variable is not set. "
            "Set it before running, e.g. `export API_KEY=...`."
        )

    # Unknown identifiers fall through unchanged so any model name your endpoint
    # serves still works (e.g. `-m gpt-4o-mini` for a quick smoke test).
    entry = MODEL_REGISTRY.get(LLM_model_name, {"model": LLM_model_name})
    upstream_name = entry["model"]
    extra_kwargs = {k: v for k, v in entry.items() if k != "model"}

    client_kwargs = {
        "model": upstream_name,
        "api_key": API_KEY,
        "temperature": temperature,
    }
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client_kwargs.update(extra_kwargs)

    return ChatOpenAI(**client_kwargs)


def test_models(test_models):
    test_message = "who are you? Introduce yourself short and concise."

    for model_name in test_models:
        try:
            start = time.time()
            _model = model(model_name, 0.3)
            response = _model.invoke([HumanMessage(content=test_message)])
            elapsed = time.time() - start

            print(f"========={model_name}=========")
            print(f"Time: {elapsed:.2f}s")
            print()

            print("--- Full Response Content ---")
            print(response.content)
            print()

            print("--- Token Information ---")
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata
                if metadata:
                    token_info = {}
                    if "token_usage" in metadata:
                        token_info = metadata["token_usage"]
                    elif "usage" in metadata:
                        token_info = metadata["usage"]
                    elif "model_extra" in metadata and isinstance(metadata["model_extra"], dict):
                        if "usage" in metadata["model_extra"]:
                            token_info = metadata["model_extra"]["usage"]

                    if token_info:
                        print(json.dumps(token_info, indent=2, ensure_ascii=False))
                    else:
                        print("Token usage info not found in response_metadata")
                        print(f"Available keys in response_metadata: {list(metadata.keys())}")
                        print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
                else:
                    print("response_metadata is empty")
            else:
                print("response_metadata attribute not found")

            if hasattr(response, "usage_metadata"):
                print(f"usage_metadata: {response.usage_metadata}")

            print()
            print("--- Raw Response Object ---")
            print(f"Type: {type(response)}")
            print(f"response: {response}")
            print()
            print("--- response_metadata (raw) ---")
            if hasattr(response, "response_metadata"):
                print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False, default=str))
            print()
            print("=" * 80)
            print()

        except Exception as e:
            print(f"========={model_name}=========")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 80)
            print()


if __name__ == "__main__":
    test_models(["gpt-4.1-mini"])
