#!/usr/bin/env python3
from llm_sdk import Small_LLM_Model


def llm_test() -> None:
    llm = Small_LLM_Model()
    prompt: str = (
        "You are a speacialist at finding correct functions for given questions.\n" + 
        "Available functions:\n ft_add_numbers(n1: int, n2: int) -> int\n" +
        " ft_greet(name: str) -> str\n" +
        "You must output ONLY the response\n" +
        "Example question: What is the sum of 2 and 3?\n" +
        "Example response: The correct function is -> ft_add_numbers(2, 3) -> int\n" +
        "Question: What is the sum of 10 and 10?\n" +
        "Response: \n"
    )
    # print(prompt)
    input_ids = llm.encode(prompt)
    ids_lst = input_ids[0].tolist()
    # print(ids_lst)
    for _ in range(20):
        logits = llm.get_logits_from_input_ids(ids_lst)
        # print(type(logits))
        next_token_id = int(logits.index(max(logits)))
        # print(next_token_id)
        ids_lst.append(next_token_id)
    result = llm.decode(ids_lst)
    print(result)
    with open("output_test.txt", "w") as test:
        test.write(str(result))


llm_test()
