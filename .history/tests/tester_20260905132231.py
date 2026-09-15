from models import FunctionCallResult, FunctionDefinition, PromptEntry
import json

# models check

# functions: list[FunctionDefinition] = []
# with open("data/input/functions_definition.json", "r") as f:
#     data = json.load(f)

# for part in data:
#     functions.append(FunctionDefinition.model_validate(part))

# for fn in functions:
#     print(fn.model_dump_json(indent=2))


# with open("data/input/function_calling_tests.json", "r") as f:
#     data = json.load(f)

# for part in data:
#     functions.append(PromptEntry.model_validate(part))

# for fn in functions:
#     print(fn.model_dump_json(indent=2))

# utils check

def 