from typing import List
from llm_sdk import Small_LLM_Model
import time
from .file_handling import Function


class LLMHandler:
    def __init__(self, functions: List[Function]) -> None:
        self.llm = Small_LLM_Model()
        self.max_tokens: int = 50
        self.elapsed: float = 0.0
        self.functions = functions

    def run_prompt(self, prompt: str) -> None:
        res_str = ""
        start = time.monotonic()
        input_tokens = self.llm.encode(prompt)
        token_ids = input_tokens[0].tolist()
        print(len(token_ids))
        for _ in range(self.max_tokens):
            logits = self.llm.get_logits_from_input_ids(token_ids)
            next_token_id = int(logits.index(max(logits)))
            token_ids.append(next_token_id)
            token = self.llm.decode(next_token_id)
            if token == "<":
                break
            res_str += token
        self.elapsed += time.monotonic() - start
        print(res_str)

    def func_name_tokens(self) -> None:
        for func in self.functions:
            func_name_tokens = self.llm.encode(func.name)
            token_ids = func_name_tokens[0].tolist()
            print(func.name)
            for token in token_ids:
                print(self.llm.decode(token))

    def param_name_tokens(self) -> None:
        for func in self.functions:
            print()
            print(func.name)
            for param_name in func.parameters:
                print(param_name, self.llm.encode(param_name))
