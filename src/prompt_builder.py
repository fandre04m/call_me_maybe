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
            "\nAvailable functions:",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Description: {func.description}")
            prompt.append("Arguments:")
            for p_name, p_type in func.parameters.items():
                prompt.append(f" {p_name} ({p_type.type})")
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
            f"\nRequest: {user_request}"
        ])
        return "\n".join(prompt)
