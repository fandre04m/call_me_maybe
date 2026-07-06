from .file_handling import Function
from typing import List, Union


class PromptBuilder:
    def sys_prompt(
        self,
        functions: List[Function],
        user_request: str
    ) -> str:
        prompt = [
            "You are a function calling system.\n",
            "Choose the correct function and parameters, based on the user prompt.",
            "Available functions:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Description: {func.description}")
            prompt.append("Arguments:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f"{p_name} ({p_type.type})")
        prompt.extend([
            "\nExample:",
            "Substitute the words 'test' in 'The test is actually just "
            "a test' with 'joke'."
            "fn_substitute_string_with_regex",
            "source_string",
            "The test is actually just a test",
            "regex",
            "test",
            "replacement",
            "joke",
            f"\nUser prompt: {user_request}"
        ])
        return "\n".join(prompt)


def to_type(p_type: str, value: str) -> Union[str, int, float]:
    if p_type == "number":
        return float(value)
    if p_type == "integer":
        return int(value)
    return str(value)
