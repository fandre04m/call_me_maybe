from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any
import json

BASE_DIR = Path(__file__).resolve().parent.parent
input_funcs = BASE_DIR / "data" / "input" / "functions_definition.json"
input_prompts = BASE_DIR / "data" / "input" / "function_calling_tests.json"


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class Prompt(BaseModel):
    prompt: str


def load_json(file_path: Path) -> List[Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_func_definitions() -> List[Function]:
    func_lst = load_json(input_funcs)

    functions: List[Function] = []

    for func in func_lst:
        functions.append(Function.model_validate(func))
    return functions


def load_prompts() -> List[Prompt]:
    prompt_lst = load_json(input_prompts)

    prompts: List[Prompt] = []

    for prompt in prompt_lst:
        prompts.append(Prompt.model_validate(prompt))
    return prompts
