from .file_handling import Function
from typing import List, Union
import json


class PromptBuilder:
    """Build prompts for constrained function call generation."""
    def make_prompt(self, functions: List[Function], user_prompt: str) -> str:
        """Construct a prompt containing available functions and user request.

        Args:
            functions: Available function definitions.
            user_prompt: User's natural language request.

        Returns:
            Prompt formatted for function call generation.
        """
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


def to_type(p_type: str, value: str) -> Union[str, int, float, bool]:
    """Convert a generated string to the requested parameter type.

    Args:
        p_type: Target parameter type.
        value: Generated value as text.

    Returns:
        Value converted to requested type.

    Raises:
        ValueError: If the value cannot be converted to the requested type.
    """
    try:
        if p_type == "string":
            try:
                return json.loads(f'"{value}"')
            except json.JSONDecodeError:
                pass
        if p_type == "number":
            return float(value)
        if p_type == "integer":
            return int(value)
        if p_type == "boolean":
            if value == "true":
                return True
            if value == "false":
                return False
            raise ValueError(f"Invalid boolean: {value!r}")
        return str(value)
    except ValueError as e:
        raise ValueError(f"Invalid {p_type}: {value!r}") from e


class GenerationLogger:
    """Log prompts and generated tokens during decoding."""
    def token(self, text: str) -> None:
        """Log generated token text without a trailing newline.

        Args:
            text: Decoded token text to display.
        """
        print(text, end="", flush=True)

    def prompt(self, text: str) -> None:
        """Log the user prompt before generation.

        Args:
            text: User prompt to display.
        """
        print(f"\nPrompt: {text}")
