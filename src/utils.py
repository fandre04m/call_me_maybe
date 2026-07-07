from .file_handling import Function
from typing import List, Union


class PromptBuilder:
    def sys_prompt(
        self,
        functions: List[Function],
        user_request: str
    ) -> str:
        prompt = [
            "You are a function selecting tool.\n",
            "Select from the list the function and arguments that better satisfy the request.",
            "Function list:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Description: {func.description}")
            prompt.append("Arguments:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f"{p_name} ({p_type.type})")
        prompt.extend([
            "\nExample:",
            # "Substitute the words 'test' in 'The test is actually just "
            # "a test' with 'joke'."
            # "fn_substitute_string_with_regex",
            # "source_string",
            # "The test is actually just a test",
            # "regex",
            # "test",
            # "replacement",
            # "joke",
            "Wave to 'John'.",
            "fn_wave",
            "name",
            "John",
            f"\nRequest: {user_request}"
        ])
        return "\n".join(prompt)

    def test_prompt(self, functions: List[Function], user_prompt: str) -> str:
        prompt = [
            "You are a function selecting tool.\n",
            "Produce only a valid JSON structure.",
            "Do not output reasoning.",
            "Function list:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Description: {func.description}")
            prompt.append("Parameters:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f"{p_name} ({p_type.type})")
        prompt.extend([
            # "Required answer format:",
            # "{",
            # '  "name": "fn_function_name",',
            # '  "parameters": {',
            # '    "param_name": "value"',
            # "  }",
            # "}",
            "",
            "Example user request:",
            "Wave to Pedro",
            "",
            "Example system answer:",
            "{",
            '  "name": "fn_wave",',
            '  "parameters": {',
            '    "name": "Pedro",',
            "  }",
            "},",
            "Select the correct function and parameters from the list.",
            f"\nRequest: {user_prompt}"
        ])
        return "\n".join(prompt)


def to_type(p_type: str, value: str) -> Union[str, int, float]:
    if p_type == "number":
        return float(value)
    if p_type == "integer":
        return int(value)
    return str(value)
