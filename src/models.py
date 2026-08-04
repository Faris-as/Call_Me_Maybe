# Pydantic for the model
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class _type(Enum):
    string = str
    number = int | float


class ParameterSpec(BaseModel):
    type: _type


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=5, max_length=100)
    description: str = Field(..., min_length=5, max_length=500)
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec


class PromptEntry(BaseModel):
    