from .file_handling import Function
from typing import List, Dict


class PromptBuilder:
    def main_prompt(
        self,
        functions: List[Function],
        user_request: str
    ) -> str:
        prompt_lines = [
            "You are a function-calling assistant.\n",
            "Select exactly one function from the following list:",
            "",
        ]
        for func in functions:
            prompt_lines.append(func.name)
        prompt_lines.append(
            f"\nRequest: {user_request}\nFunction name = ",
        )
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
            "You are assigning a value to a parameter of a function.",
            # "Look for possible values or hints in the user request.",
            "Prioritize something present in the user request."
            f"\nSelected function: {function.name}",
            f"Function purpose: {function.description}",
            "Function parameters:",
        ]
        for name, p_type in function.parameters.items():
            # if name in already_filled:
            #     prompt_lines.append(
            #         f"{name} ({p_type.type}) = {already_filled[name]}"
            #     )
            # else:
            prompt_lines.append(f"{name} ({p_type.type})")
        if param_name == "replacement":
            prompt_lines.append(
                "\nSelect in the request something that could be the replacement.\n"
                "Prioritize single words or symbols, usually after the word 'with'.\n"
            )
        if param_name == "regex":
            prompt_lines.append(
                "\nIf the request asks to Replace or Substitute, "
                "the answer is a valid Regular Expression that would extract the "
                "correct characters from source_string.\n"
                "Examples:\n"
                "Vowels: [aeiouAEIOU]\n"
                "Numbers: [0-9]+\n"
                "Extract the word 'test': test\n"
            )
        prompt_lines.append(f"User request: {user_request}")
        prompt_lines.append(f"Value for {param_name}: ")
        return "\n".join(prompt_lines)
