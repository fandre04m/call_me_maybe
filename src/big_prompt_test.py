from llm_sdk import Small_LLM_Model


def test_generator(prompt: str) -> None:
    max_tokens = 15
    llm = Small_LLM_Model()
    generated = []

    input_ids = llm.encode(prompt)[0].tolist()
    while len(generated) < max_tokens:
        logits = llm.get_logits_from_input_ids(input_ids)
        next_token = logits.index(max(logits))
        print(llm.decode([next_token]), end="")
        input_ids.append(next_token)
