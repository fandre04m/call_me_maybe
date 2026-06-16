from src import Function
from typing import List


class PromptBuilder:
    def build(
        self,
        functions: List[Function],
    ) -> str:
        prompt = [
            "You are a function-calling assistant.",
            "Select one function and output its arguments.",
            "",
            "Functions:",
        ]
        for func in functions:
            prompt.append(func.name)
            for name, param in func.parameters.items():
                prompt.append(f"{name} ({param.type})")
        prompt.extend([
            "",
            "Output format (one block per argument, ~ as separator):",
            "FUNCTION_NAME",
            "ARG_NAME",
            "ARG_VALUE",
            "~",
            "ARG_NAME",
            "ARG_VALUE",
            "~",
            "",
            "Example:",
            "Request: What is the sum of 8 and 9?",
            "fn_add_numbers",
            "a",
            "8",
            "~",
            "b",
            "9",
            "~",
            # "",
            # "Example:",
            # "Request: Greet pedro",
            # "fn_greet",
            # "name",
            # "pedro",
            # "~",
            # "",
            # "Example:",
            # "Request: Replace all vowels in 'hello world' with underscores.",
            # "fn_substitute_string_with_regex",
            # "source_string",
            # "hello world",
            # "~",
            # "regex",
            # "[^aeiou]",
            # "~",
            # "replacement",
            # "_",
            # "~",
            "",
            "Request: ",
        ])
        return "\n".join(prompt)
