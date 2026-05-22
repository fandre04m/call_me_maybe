from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Union
import json

BASE_DIR = Path(__file__).resolve().parent.parent
input_funcs = BASE_DIR / "data" / "input" / "functions_definition.json"
input_prompts = BASE_DIR / "data" / "input" / "function_calling_tests.json"
output_path = BASE_DIR / "data" / "output" / "function_calling_results.json"


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


def write_call_result() -> None:
    mock_results = [
        CallResult(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2, "b": 3}
        ),
        CallResult(
            prompt="Reverse the string 'hello'",
            name="fn_reverse_string",
            parameters={"s": "hello"}
        )
    ]
    data = []
    for res in mock_results:
        data.append(res.model_dump())
    print(data)
