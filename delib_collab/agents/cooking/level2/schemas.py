#!/usr/bin/env python
# -*- coding: UTF-8 -*-


from pydantic import BaseModel, Field
from typing import Dict, Any, List,Optional

class Step1(BaseModel):
    thinking_process: Optional[str] = Field(None, description="The thinking process involved in step 1")
    partner_ingredients: Optional[Dict[str, int]] = Field(None, description="Ingredients provided by the partner")
    partner_preference: Optional[Dict[str, float]] = Field(None, description="Partner's preference for dishes")

class Step2(BaseModel):
    thinking_process: Optional[str] = Field(None, description="The thinking process involved in step 2")
    total_ingredients: Optional[Dict[str, int]] = Field(None, description="Total ingredients used in the process")
    overall_preference: Optional[Dict[str, float]] = Field(None, description="Overall preference for dishes")

class ObseverFormat(BaseModel):
    step_1: Optional[Step1] = Field(None, description="Step 1 of the process")
    step_2: Optional[Step2] = Field(None, description="Step 2 of the process")

    class Config:
        json_schema_extra = {
            "required": ["step_1", "step_2"]
        }


class ThinkingFormat(BaseModel):
    reasoning_process: str = Field(..., description="The reasoning process for the menu proposal")
    menu_proposal: List[str] = Field(..., description="A list of proposed dishes")
    explanation: str = Field(..., description="Explanation for the menu proposal")


class DecisionProcessFormat(BaseModel):
    reasoning_process: str = Field(..., description="The reasoning process behind the decision")
    accept_decision: bool = Field(..., description="Whether the decision is accepted (true/false)")
    explanation: str = Field(..., description="Explanation for the decision")

class MessageFormat(BaseModel):
    thinking_process: str = Field(..., description="The reasoning or thinking process involved")
    message_content: str = Field(..., description="The content of the message")

