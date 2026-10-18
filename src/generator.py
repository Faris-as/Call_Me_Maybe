# the actual token-by-token loop using llm_sdk
import json
import math
from llm_sdk import Small_LLM_Model
from json_fsm import JsonFSM
from models import FunctionDefinition

def build_system_prompt(funcitons: list[FunctionDefinition]) -> str:
    func_descriptions = []
    for fn in funcitons:
        params