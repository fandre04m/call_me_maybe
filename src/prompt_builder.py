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
            "Your task is to select exactly one function",
            "and produce ONLY valid JSON.",
            "",
            "Do not explain anything.",
            "Do not output prose.",
            "Do not output markdown.",
            "Output JSON only.",
            "",
            "Available functions:",
            "",
        ]
        for func in functions:
            prompt.append(f"Function: {func.name}")
            prompt.append(f"Description: {func.description}")
            prompt.append("Arguments:")
            for name, param in func.parameters.items():
                prompt.append(f" - {name} ({param.type})")
            prompt.append("")
        prompt.extend([
            "Required answer format:",
            "{",
            '  "name": "function_name",',
            '  "parameters": {',
            '    "arg_name": "value"',
            "  }",
            "}",
            "",
            "Example user request:",
            "What is the sum of 4 and 7?",
            "",
            "Example answer:",
            "{",
            '  "name": "ft_add_numbers",',
            '  "parameters": {',
            '    "a": "4",',
            '    "b": "7"',
            "  }",
            "},",
            "",
            "User request:",
            user_prompt
        ])
        return "\n".join(prompt)
