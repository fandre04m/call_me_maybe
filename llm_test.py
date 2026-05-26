#!/usr/bin/env python3
from llm_sdk import Small_LLM_Model


def llm_test() -> None:
    llm = Small_LLM_Model()
    prompt: str = "A person named John"
    print(prompt)
    input_ids = llm.encode(prompt)
    ids_lst = input_ids[0].tolist()
    # print(ids_lst)
    for _ in range(20):
        logits = llm.get_logits_from_input_ids(ids_lst)
        # print(type(logits))
        next_token_id = int(logits.index(max(logits)))
        # print(next_token_id)
        ids_lst.append(next_token_id)
    next_token_added = llm.decode(ids_lst)
    print(next_token_added)


llm_test()
