from .file_handling import Function
from typing import List


class PromptBuilder:
    def build(
        self,
        functions: List[Function],
        user_prompt: str
    ) -> str:
        prompt = [
            "You are a system that calls the correct function,",
            "with the correct arguments, based on a given prompt.",
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
            "You must output ONLY valid JSON",
            "",
            "Required format:",
            "{",
            '  "name": "function_name",',
            '  "parameters": {',
            '    "arg_name": "value"',
            "  }",
            "}",
            "",
            "User request:",
            user_prompt
        ])
        return "\n".join(prompt)
