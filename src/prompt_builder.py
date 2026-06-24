from .file_handling import Function
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
        param_type: str,
        already_filled: Dict[str, str],
    ) -> str:
        prompt_lines = [
            "You are a tool to determine parameter values for a function.",
            "The values needed are based on the user request.",
            f"\nSelected function:\n{function.name}: {function.description}",
            "Parameters and their type:",
        ]
        for name, p_type in function.parameters.items():
            if name in already_filled:
                prompt_lines.append(
                    f"{name} ({p_type.type}) = {already_filled[name]}"
                )
            else:
                prompt_lines.append(f"{name} ({p_type.type}) =")
        prompt_lines.append(f"\nUser request: {user_request}")
        if param_name == "replacement":
            prompt_lines.append(
                "\nSelect in the request something that could be the "
                "replacement. Prefer something really short, "
                "like a single word."
            )
        if param_name == "regex":
            prompt_lines.append(
                "\nOutput a valid Regular Expression."
            )
        prompt_lines.append(f"\n{param_name} ({param_type}) = ")
        return "\n".join(prompt_lines)
