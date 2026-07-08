from .file_handling import Function
from typing import List, Union
import json


class PromptBuilder:
    def make_prompt(self, functions: List[Function], user_prompt: str) -> str:
        prompt = [
            "You are a function calling tool.\n",
            "Produce ONLY valid JSON.",
            "Select the correct function and parameters from the list:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append("Parameters:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f"{p_name} ({p_type.type})")
        prompt.extend([
            "",
            "Example:",
            "What is the result of 4 + 8?",
            "{",
            '  "name": "fn_add_numbers",',
            '  "parameters": {',
            '    "a": 4,',
            '    "b": 8',
            "  }",
            "}",
            "",
            f"\n{user_prompt}"
        ])
        return "\n".join(prompt)


def to_type(p_type: str, value: str) -> Union[str, int, float]:
    if p_type == "string":
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            pass
    if p_type == "number":
        return float(value)
    if p_type == "integer":
        return int(value)
    return str(value)
