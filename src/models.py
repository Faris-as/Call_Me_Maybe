# Pydantic for the model
from pydantic import BaseModel, Field
from typing import Any
from enum import Enum


class ParamType(str, Enum):
    string = "string"
    number = "number"


class ParameterSpec(BaseModel):
    type: ParamType


class FunctionDefinition(BaseModel):
    name: str = Field(...)
    description: str = Field(..., min_length=5, max_length=500)
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptEntry(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str = Field(..., min_length=5, max_length=100)
    parameters: dict[str, Any]
