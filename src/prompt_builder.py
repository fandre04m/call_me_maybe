from src import Function
from typing import List, Dict


class PromptBuilder:
    def main_prompt(
        self,
        functions: List[Function],
    ) -> str:
        prompt_lines = [
            "You are a function-calling assistant.",
            "Select one function and output its arguments.",
            "",
            "Functions:",
        ]
        for func in functions:
            prompt_lines.append(f" {func.name}")
            for name in func.parameters.keys():
                prompt_lines.append(f"  {name}")
        prompt_lines.extend([
            "",
            "Output format:",
            "FUNCTION_NAME",
            "ARG_NAME",
            "ARG_NAME",
            "",
            "Example:",
            "Request: What is the sum of 8 and 9?",
            "fn_add_numbers",
            "a",
            "b",
            "",
            "Request: ",
        ])
        return "\n".join(prompt_lines)

    def value_prompt(
        self,
        user_request: str,
        function: Function,
        param_name: str,
        already_filled: Dict[str, str]
    ) -> str:
        prompt_lines = [
            "You are a tool to determine parameter values.",
            "Select only one value for the desired parameter, nothing else.\n"
            f" {user_request}",
            f"Selected function: {function.name}",
            "Required function parameters:"
        ]
        for name in function.parameters.keys():
            prompt_lines.append(f"{name}")
        if already_filled:
            prompt_lines.append("\nParameter values already determined:")
            for name, val in already_filled.items():
                prompt_lines.append(f"{name} = {val}")
        prompt_lines.append("\nNext value to determine:")
        prompt_lines.append(f"{param_name} = ")
        return "\n".join(prompt_lines)
