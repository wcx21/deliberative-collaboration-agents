#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""LLM API configuration.

All models are served through an OpenAI-compatible endpoint:

- ``API_KEY``      — credential for that endpoint (required).
- ``API_BASE_URL`` — base URL of the endpoint (optional). Leave it unset to use
  OpenAI's default endpoint, or point it at any OpenAI-compatible gateway
  (e.g. a self-hosted vLLM server or an aggregating proxy) that serves the
  model names you request.
"""

import os


API_KEY = os.environ.get("API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
