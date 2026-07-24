#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Task allocation Level 1 response format definitions."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional


# ================ OBSERVE format ================

class ObserverStep1(BaseModel):
    """Step 1: Infer partner resources and preferences."""
    model_config = ConfigDict(extra='forbid')
    
    thinking_process: str = Field(default="", description="The thinking process involved in step 1")
    partner_resources: str = Field(
        default="", 
        description="Resources observed or inferred from partners as JSON string. Format: {'agent_X': {'private_resources': {...}, 'public_resources': {...}}}"
    )
    partner_preferences: str = Field(
        default="",
        description="Partners' preferences for tasks as JSON string. Format: {'agent_X': {'task_1': 0.8, 'task_2': 0.6, ...}}"
    )


class ObserverStep2(BaseModel):
    """Step 2: Merge global resource state and preferences."""
    model_config = ConfigDict(extra='forbid')
    
    thinking_process: str = Field(default="", description="The thinking process involved in step 2")
    total_resources: str = Field(
        default="",
        description="Total resources after merging all agents' observations as JSON string. Format: {'agent_private_resources': {'agent_0': {...}, 'agent_1': {...}, 'agent_2': {...}}, 'public_resources': {...}}"
    )
    overall_preferences: str = Field(
        default="",
        description="Overall preferences for tasks after merging as JSON string. Format: {'agent_0': {'task_1': 0.8, ...}, 'agent_1': {...}, 'agent_2': {...}}"
    )


class ObserverFormat(BaseModel):
    """Observer format containing two-step observation results.

    Defines both nested (step_1/step_2) and flattened fields. In practice,
    auto_parse_text extracts the flattened fields from LLM XML-tagged output.
    """
    model_config = ConfigDict(extra='forbid')
    
    step_1: ObserverStep1 = Field(default_factory=ObserverStep1, description="Step 1: Infer partner resources and preferences")
    step_2: ObserverStep2 = Field(default_factory=ObserverStep2, description="Step 2: Merge total resources and preferences")
    
    
    partner_resources: str = Field(
        default="",
        description="Partner resources (flattened, for direct access) as JSON string. Format: {'agent_X': {'private_resources': {...}, 'public_resources': {...}}}"
    )
    partner_preferences: str = Field(
        default="",
        description="Partner preferences (flattened, for direct access) as JSON string. Format: {'agent_X': {'task_1': 0.8, ...}}"
    )
    total_resources: str = Field(
        default="",
        description="Total resources (flattened, for direct access) as JSON string. Format: {'agent_private_resources': {...}, 'public_resources': {...}}"
    )
    overall_preferences: str = Field(
        default="",
        description="Overall preferences (flattened, for direct access) as JSON string. Format: {'agent_0': {...}, 'agent_1': {...}, 'agent_2': {...}}"
    )


# ================ THINK format ================

class ThinkingFormat(BaseModel):
    """Thinking format: propose a task allocation."""
    model_config = ConfigDict(extra='forbid')
    
    reasoning_process: str = Field(..., description="The reasoning process for the task allocation proposal")
    allocation_proposal: str = Field(
        ...,
        description="A JSON string mapping tasks to agents. Format: {'task_1': 'agent_0', 'task_2': 'agent_1', ...}. Each task must be assigned to exactly one agent."
    )
    explanation: str = Field(..., description="Explanation for the allocation proposal")


# ================ DECISION format ================

class DecisionProcessFormat(BaseModel):
    """Decision format: accept or reject a partner's proposal."""
    model_config = ConfigDict(extra='forbid')
    
    reasoning_process: str = Field(..., description="The reasoning process behind the decision")
    accept_decision: bool = Field(..., description="Whether the decision is accepted (true/false)")
    explanation: str = Field(..., description="Explanation for the decision")
    
    
    final_decision: str = Field(
        default="",
        description="Final allocation decision as JSON string (only for Leader). Format: {'task_1': 'agent_0', ...}. If provided, this overrides accept_decision."
    )


# ================ MESSAGE format ================

class MessageFormat(BaseModel):
    """Message format: generate a conversation message."""
    model_config = ConfigDict(extra='forbid')
    
    thinking_process: str = Field(..., description="The reasoning or thinking process involved")
    message_content: str = Field(..., description="The content of the message to send to partners")

