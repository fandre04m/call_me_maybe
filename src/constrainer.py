from llm_sdk import Small_LLM_Model
import time


class ModelCaller:
    def __init__(self) -> None:
        self.llm = Small_LLM_Model()
        self.max_tokens: int = 100
        self.elapsed: float = 0.0

    def run_prompt(self, prompt: str) -> None:
        res_str = ""
        input_ids = self.llm.encode(prompt)
        ids_lst = input_ids[0].tolist()
        print(len(ids_lst))
        for _ in range(self.max_tokens):
            start = time.monotonic()
            logits = self.llm.get_logits_from_input_ids(ids_lst)
            self.elapsed += time.monotonic() - start
            print(self.elapsed)
            next_token_id = int(logits.index(max(logits)))
            ids_lst.append(next_token_id)
            token = self.llm.decode(next_token_id)
            if token == "<":
                break
            res_str += token
        print(res_str)
