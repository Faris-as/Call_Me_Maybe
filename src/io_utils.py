"""Load and save the project's JSON files, raising InputError on failure."""
import json
import os
from typing import Any

from pydantic import ValidationError

from .models import FunctionCallResult, FunctionDefinition, PromptEntry


class InputError(Exception):
    """Raised when an input or output file cannot be used."""


def _load_json_list(file_path: str, label: str) -> list[Any]:

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
    except FileNotFoundError:
        raise InputError(f"{label}: file not found: {file_path}") from None
    except UnicodeDecodeError:
        raise InputError(f"{label}: not valid UTF-8: {file_path}") from None
    except OSError as e:
        raise InputError(f"{label}: cannot read {file_path}: {e}") from None

    if not raw.strip():
        raise InputError(f"{label}: file is empty: {file_path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise InputError(
            f"{label}: invalid JSON in {file_path}: {e}"
        ) from None

    if not isinstance(data, list):
        raise InputError(
            f"{label}: expected a JSON list, got {type(data).__name__}"
        )
    return data


def read_func_definition(file_path: str) -> list[FunctionDefinition]:
    """Return validated function definitions (at least one, unique names)."""
    result: list[FunctionDefinition] = []
    for i, entry in enumerate(_load_json_list(file_path, "functions")):
        try:
            result.append(FunctionDefinition.model_validate(entry))
        except ValidationError as e:
            raise InputError(
                f"functions: entry #{i} is invalid: {e}"
            ) from None
    if not result:
        raise InputError("functions: the list is empty")
    names = [f.name for f in result]
    if len(set(names)) != len(names):
        raise InputError("functions: duplicate function names")
    return result


def read_prompts(file_path: str) -> list[PromptEntry]:
    """Return validated prompts. An empty list is allowed."""
    result: list[PromptEntry] = []
    for i, entry in enumerate(_load_json_list(file_path, "prompts")):
        try:
            result.append(PromptEntry.model_validate(entry))
        except ValidationError as e:
            raise InputError(
                f"prompts: entry #{i} is invalid: {e}"
            ) from None
    return result


def save_results(results: list[FunctionCallResult], file_path: str) -> None:
    """Write results as JSON: temp file first, then swap it in."""
    tmp_path = file_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f,
                      indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except OSError as e:
        raise InputError(f"cannot write output {file_path}: {e}") from None