from llm_sdk import Small_LLM_Model


class ModelCaller:
    def run_prompt(self, prompt: str) -> None:
        llm = Small_LLM_Model()
        input_ids = llm.encode(prompt)
        ids_lst = input_ids[0].tolist()
        for _ in range(70):
            logits = llm.get_logits_from_input_ids(ids_lst)
            next_token_id = int(logits.index(max(logits)))
            ids_lst.append(next_token_id)
        result = llm.decode(ids_lst[(len(input_ids[0].tolist())) + 4:])
        print(result)
