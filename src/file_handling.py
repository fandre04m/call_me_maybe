from pathlib import Path
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Dict, Union, Literal
import json


class EmptyDataError(Exception):
    """Raised when a required JSON file contains no entries."""


class Parameter(BaseModel):
    """Schema describing a function parameter type."""
    type: Literal["string", "number", "integer", "boolean"]


class Function(BaseModel):
    """Schema describing a callable function and its interface."""
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]

    @field_validator("name", "description")
    @classmethod
    def not_empty(cls, value: str) -> str:
        """Validate that a string field is not empty.

        Args:
            value: String to validate.

        Returns:
            Validated string.

        Raises:
            ValueError: If the string is empty or contains only whitespace.
        """
        if not value.strip():
            raise ValueError("Function name/description cannot be empty.")
        return value

    @field_validator("parameters")
    @classmethod
    def no_param_name(
        cls,
        value: Dict[str, Parameter]
    ) -> Dict[str, Parameter]:
        """Validate that all parameter names are non-empty.

        Args:
            value: Mapping of parameter names to parameter definitions.

        Returns:
            Validated parameter mapping.

        Raises:
            ValueError: If any parameter name is empty or contains only
                whitespace.
        """
        for name in value:
            if not name.strip():
                raise ValueError("Parameter name cannot be empty.")
        return value


class Prompt(BaseModel):
    """Schema representing a user prompt."""
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate that a prompt is not empty.

        Args:
            value: Prompt text to validate.

        Returns:
            Validated prompt.

        Raises:
            ValueError: If the prompt is empty or contains only whitespace.
        """
        if not value.strip():
            raise ValueError("Prompt cannot be empty.")
        return value


class CallResult(BaseModel):
    """Result of generated function call."""
    prompt: str
    name: str
    parameters: Dict[str, Union[str, int, float, bool]]


def load_json(file_path: Path) -> List[Dict[str, str]]:
    """Load JSON data from a file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, str]] = json.load(f)
        return data


class FileLoader:
    """Load input data and write generated function call results."""
    def __init__(self) -> None:
        self.func_definitions: List[Function] = []
        self.prompts: List[Prompt] = []

    def load_func_definitions(self, functions_path: Path) -> None:
        """Load and validate function definitions from a JSON file.

        Invalid definitions are skipped. A fallback ``fn_none_found`` function
        is always added.

        Args:
            functions_path: Path to function definitions file.
        """
        func_lst = load_json(functions_path)
        if not func_lst:
            raise EmptyDataError("Function definitions file is empty.")

        for i, func in enumerate(func_lst):
            try:
                self.func_definitions.append(Function.model_validate(func))
            except ValidationError as e:
                print(f"Skipping function #{i}: {e.errors()[0]['msg']}\n")
        none_func = Function(
            name='fn_none_found',
            description="Function to select when no good option is available.",
            parameters={},
            returns={
                "type": "none"
            }
        )
        self.func_definitions.append(Function.model_validate(none_func))

    def load_prompts(self, input_path: Path) -> None:
        """Load and validate user prompts from a JSON file.

        Invalid prompts are skipped.

        Args:
            input_path: Path to prompt file.
        """
        prompt_lst = load_json(input_path)
        if not prompt_lst:
            raise EmptyDataError("Function calling tests file is empty.")

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
        """Write generated function call results to a JSON file.

        Args:
            call_res: Generated function call results.
            output_path: Destination path for output JSON file.
        """
        data = []
        for res in call_res:
            data.append(res.model_dump())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
