# Pydantic for the model
from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from enum import Enum


class ParamType(str, Enum):
    string = "string"
    number = "number"
    integer = "integer"
    boolean = "boolean"

class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: ParamType


class FunctionDefinition(BaseModel):

    model_config = ConfigDict(extra="ignore")
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ParameterSpec | None = None


class PromptEntry(BaseModel):
    
    model_config = ConfigDict(extra="ignore")
    prompt: str


class FunctionCallResult(BaseModel):

    model_config = ConfigDict(extra="ignore")
    prompt: str
    name: str
    parameters: dict[str, Any]
