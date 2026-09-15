# load/save JSON with try/except, clear error messages
import json
from pydantic import ValidationError
from models import FunctionDefinition, PromptEntry


def read_func_definition(file_path: str) -> list[FunctionDefinition]:
    try:
        with open(file_path, "r") as f:
            raw = f.read()

        if not raw.strip():
            print(f"Error: file is empty: {file_path}")
        
        data = json.loads(f)

        if not isinstance(data, list):
            print(f"Error: expected a JSON list, got {type(data).__name__}")
            return []
        
        result: list[FunctionDefinition] = []
        for i, entry in enumerate(data):
            result.append(FunctionDefinition.model_vald)

    except FileNotFoundError:
        print(f"Error: file not found: {file_path}")
        return []

    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {file_path}: {e}")
        return []

    except ValidationError as e:
        print(f"Error:invalid FunctionDefintion data: {e}")
        return []

    except OSError as e:
        print(f"Error: could not read file {file_path}: {e}")
        return []

    except Exception as e:
        print(f"Error: unexpected error while loading {file_path}: {e}")
        return []
