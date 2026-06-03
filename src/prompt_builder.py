from .file_handling import Function
from typing import List


class PromptBuilder:
    def build(
        self,
        functions: List[Function],
        user_prompt: str
    ) -> str:
        prompt = [
            "You are function calling system.",
            "",
            "Your task is to select exactly one function ",
            "and it's required arguments.",
            # "and produce ONLY valid JSON.",
            "",
            "Do not output reasoning.",
            "Do not output prose.",
            "Do not output markdown.",
            "",
            "Available functions:",
            "",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            # prompt.append(f"Description: {func.description}")
            prompt.append("Arguments:")
            for name, param in func.parameters.items():
                prompt.append(f" - {name} ({param.type})")
            prompt.append("")
        prompt.extend([
            "Required answer format:",
            "",
            "fn_function_name",
            '"arg_name": value',
            # "{",
            # '  "name": "fn_function_name",',
            # '  "parameters": {',
            # '    "arg_name": "value"',
            # "  }",
            # "}",
            "<END>",
            "",
            "Example user request:",
            "What is the sum of 4 and 7?",
            "",
            "Example answer:",
            "fn_add_numbers",
            '"a": 4',
            '"b": 7',
            # "<END>",
            # "",
            # "Example user request:",
            # "Reverse the string 'hello'",
            # "",
            # "Example answer:",
            # "fn_reverse_string",
            # '"s": "hello"'
            # "{",
            # '  "name": "fn_add_numbers",',
            # '  "parameters": {',
            # '    "a": "4",',
            # '    "b": "7"',
            # "  }",
            # "},",
            "<END>",
            "",
            "User request:",
            user_prompt
        ])
        return "\n".join(prompt)
