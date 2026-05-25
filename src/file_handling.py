from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Union
import json


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class Prompt(BaseModel):
    prompt: str


class CallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Union[str, int, float, bool]]


def load_json(file_path: Path) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_func_definitions(functions_path: Path) -> List[Function]:
    func_lst = load_json(functions_path)

    functions: List[Function] = []

    for func in func_lst:
        functions.append(Function.model_validate(func))
    return functions


def load_prompts(input_path: Path) -> List[Prompt]:
    prompt_lst = load_json(input_path)

    prompts: List[Prompt] = []

    for prompt in prompt_lst:
        prompts.append(Prompt.model_validate(prompt))
    return prompts


def write_call_result(call_res: List[CallResult], output_path: Path) -> None:
    data = []
    for res in call_res:
        data.append(res.model_dump())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
