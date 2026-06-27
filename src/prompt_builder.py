from .file_handling import Function
from typing import List


class PromptBuilder:
    def sys_prompt(
        self,
        functions: List[Function],
        user_request: str
    ) -> str:
        prompt = [
            "Select one function that fullfils the request.\n",
            # "Remove any quotes from words in the answer.",
            "\nAvailable functions:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Function purpose: {func.description}")
            prompt.append("Arguments:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f" {p_name} ({p_type.type})")
        prompt.extend([
            # "\nRequired format:",
            # "fn_function_name",
            # "arg_name",
            # "arg_value",
            # "arg_name",
            # "arg_value",
            # "...",
            "\nExamples:",
            "What is the sum of 4 and 7?",
            "fn_add_numbers",
            "a",
            "4",
            "b",
            "7",
            "\nReverse the word 'crying'",
            "fn_reverse_string",
            "s",
            "crying",
            f"\nRequest: {user_request}"
        ])
        return "\n".join(prompt)
