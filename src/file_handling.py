from pathlib import Path
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict, Union
import json


class Parameter(BaseModel):
    """
    Parameter structure validation
    """
    type: str


class Function(BaseModel):
    """
    Function structure validation
    """
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]


class Prompt(BaseModel):
    """
    Prompt structure validation.
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty.")
        return value


class CallResult(BaseModel):
    """
    Result structure validation.
    """
    prompt: str
    name: str
    parameters: Dict[str, Union[str, int, float, bool]]


def load_json(file_path: Path) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class FileLoader:
    """
    Class to load and validate the JSON files
    and to write the validated output.
    """
    def __init__(self) -> None:
        self.func_definitions: List[Function] = []
        self.prompts: List[Prompt] = []

    def load_func_definitions(self, functions_path: Path) -> None:
        func_lst = load_json(functions_path)
        for func in func_lst:
            self.func_definitions.append(Function.model_validate(func))
        none_func = Function(
            name='no_func_found',
            description="Function to select when not able "
                        "to find a good match.",
            parameters={},
            returns={
                "type": "none"
            }
        )
        self.func_definitions.append(Function.model_validate(none_func))

    def load_prompts(self, input_path: Path) -> None:
        prompt_lst = load_json(input_path)
        for i, prompt in enumerate(prompt_lst):
            try:
                self.prompts.append(Prompt.model_validate(prompt))
            except ValidationError as e:
                print(f"Skipping prompt #{i}: {e.errors()[0]['msg']}\n")

    def write_call_result(
        self,
        call_res: List[CallResult],
        output_path: Path
    ) -> None:
        data = []
        for res in call_res:
            data.append(res.model_dump())
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
