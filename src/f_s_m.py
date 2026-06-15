from typing import Dict, List, Set
from enum import Enum
from llm_sdk import Small_LLM_Model
from src import Function, PromptBuilder
import time


class TrieNode():
    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = {}
        self.name: str | None = None


class PrefixTrie():
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, name_ids: List[int], name: str) -> None:
        current = self.root

        for token_id in name_ids:
            if token_id not in current.children:
                current.children[token_id] = TrieNode()
            current = current.children[token_id]

        current.name = name

    def _get_node(self, prefix: List[int]) -> TrieNode | None:
        current = self.root

        for token_id in prefix:
            if token_id not in current.children:
                return None

            current = current.children[token_id]

        return current

    def allowed_tokens(self, prefix: List[int]) -> Set[int]:
        node: TrieNode | None = self._get_node(prefix)

        if node is None:
            return set()

        return set(node.children.keys())

    def get_name(self, prefix: List[int]) -> str | None:
        node: TrieNode | None = self._get_node(prefix)

        if node is None:
            return None

        return node.name

    def is_complete(self, prefix: List[int]) -> bool:
        return self.get_name(prefix) is not None


class State(Enum):
    SELECT_FUNCTION = 1
    SELECT_PARAM = 2
    SELECT_VALUE = 3
    DONE = 4


class GeneratorFSM():
    def __init__(self, functions: List[Function]) -> None:
        self.max_tokens = 5
        self.llm = Small_LLM_Model()
        self.func_trie = PrefixTrie()
        self.functions = functions
        self.elapsed_time: float = 0.0
        self.state = State.SELECT_FUNCTION
        self.curr_func: Function
        self.remaining_params = []
        self.input_ids = []

        for func in functions:
            token_ids = self.llm.encode(func.name)[0].tolist()
            self.func_trie.insert(token_ids, func.name)

    def _mask_logits(
        self,
        logits: List[float],
        allowed: Set[int]
    ) -> List[float]:
        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = float("-inf")
        return logits

    def gen_func_name(self, prompt: str) -> str:
        generated: List[int] = []

        start = time.monotonic()
        self.input_ids = self.llm.encode(prompt)[0].tolist()

        while not self.func_trie.is_complete(generated):
            allowed = self.func_trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = self._mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)

        if self.func_trie.get_name(generated) == "fn_no_match":
            raise ValueError(
                "Could not find a valid function for the prompt."
            )

        self.elapsed_time += time.monotonic() - start
        return self.llm.decode(generated)

    def gen_param_name(self) -> str:
        param_trie = PrefixTrie()

        for param in self.remaining_params:
            token_ids = self.llm.encode(param)[0].tolist()
            param_trie.insert(token_ids, param)

        generated: List[int] = []
        start = time.monotonic()
        while not param_trie.is_complete(generated):
            allowed = param_trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = self._mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)

        self.elapsed_time += time.monotonic() - start
        self.remaining_params.remove(self.llm.decode(generated))
        return self.llm.decode(generated)

    def gen_param_val(self) -> str:
        generated = []

        start = time.monotonic()
        while len(generated) < self.max_tokens:
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            next_token = logits.index(max(logits))
            print(next_token)
            print(self.llm.decode([next_token]))
            # if next_token == 27:
            #     break
            self.input_ids.append(next_token)
            generated.append(next_token)

        self.elapsed_time += time.monotonic() - start
        return self.llm.decode(generated)

    def run(self, user_prompt: str) -> None:
        fixed_prompt = PromptBuilder()
        prompt = fixed_prompt.build(self.functions) + user_prompt
        print(user_prompt)

        func_name = self.gen_func_name(prompt)
        print(func_name)
        self.state = State.SELECT_PARAM

        for func in self.functions:
            if func.name == func_name:
                self.curr_func = func
                self.remaining_params = list(func.parameters.keys())

        while self.remaining_params:
            param_name = self.gen_param_name()
            print(param_name)
            self.state = State.SELECT_VALUE
            # param_type = self.curr_func.parameters[param_name].type
            # print(param_type)
            param_value = self.gen_param_val()
            print(param_value)
            # if param_type == "number" or param_type == "float":
            #     pass
            # elif param_type == "string":
            #     pass
            # else:
            #     raise ValueError(
            #         "Invalid parameter type."
            #     )
        self.state = State.SELECT_PARAM

        if not self.remaining_params:
            self.state = State.DONE
