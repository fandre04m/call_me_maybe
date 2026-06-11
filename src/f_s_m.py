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

    def get_node(self, prefix: List[int]) -> TrieNode | None:
        current = self.root

        for token_id in prefix:
            if token_id not in current.children:
                return None

            current = current.children[token_id]

        return current

    def allowed_tokens(self, prefix: List[int]) -> Set[int]:
        node: TrieNode | None = self.get_node(prefix)

        if node is None:
            return set()

        return set(node.children.keys())

    def get_name(self, prefix: List[int]) -> str | None:
        node: TrieNode | None = self.get_node(prefix)

        if node is None:
            return None

        return node.name

    def is_complete(self, prefix: List[int]) -> bool:
        return self.get_name(prefix) is not None


class State(Enum):
    SELECT_FUNCTION = 1
    SELECT_PARAMS = 2
    DONE = 3


class GeneratorFSM():
    def __init__(self, functions: List[Function]) -> None:
        self.llm = Small_LLM_Model()
        self.functions = functions
        self.elapsed_time: float = 0.0
        self.state = State.SELECT_FUNCTION
        self.func_name = ""
        self.input_ids = []
        self.func_trie = PrefixTrie()

        for func in functions:
            token_ids = self.llm.encode(func.name)[0].tolist()
            self.func_trie.insert(token_ids, func.name)

    def mask_logits(
        self,
        logits: List[float],
        allowed: Set[int]
    ) -> List[float]:
        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = float("-inf")
        return logits

    def gen_func_name(self, user_prompt: str) -> str:
        fixed_prompt = PromptBuilder()
        prompt = fixed_prompt.build(self.functions) + user_prompt
        print(user_prompt)
        generated: List[int] = []

        start = time.monotonic()
        input_ids = self.llm.encode(prompt)[0].tolist()

        while not self.func_trie.is_complete(generated):
            allowed = self.func_trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(input_ids)
            masked_logits = self.mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            input_ids.append(next_token)
            generated.append(next_token)

        self.elapsed_time += time.monotonic() - start
        self.input_ids = input_ids
        return self.llm.decode(generated)

    def gen_param_names(self, func_name: str) -> None:
        params = {}
        for func in self.functions:
            if func.name == func_name:
                params = func.parameters

        param_trie = PrefixTrie()
        for param_name in params:
            token_ids = self.llm.encode(param_name)[0].tolist()
            param_trie.insert(token_ids, param_name)

        print(func_name)
        print(param_trie.allowed_tokens([]))

    def run(self, user_prompt: str) -> None:
        self.func_name = self.gen_func_name(user_prompt)
        self.state = State.SELECT_PARAMS

        self.gen_param_names(self.func_name)
        self.state = State.DONE
