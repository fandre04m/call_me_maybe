from typing import Dict, List, Set
from enum import Enum
from llm_sdk import Small_LLM_Model
from src import Function


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


class Estate(Enum):
    SELECT_FUNCTION = 1
    DONE = 2


class GeneratorFSM():
    def __init__(self, functions: List[Function]) -> None:
        self.llm = Small_LLM_Model()
        self.elapsed_time: float = 0.0
        self.state = Estate.SELECT_FUNCTION
        self.func_trie = PrefixTrie()

        for func in functions:
            token_ids = self.llm.encode(func.name)[0].tolist()
            self.func_trie.insert(token_ids, func.name)

    def gen_func_name(self) -> None:
        generated: List[int] = []

        while not self.func_trie.is_complete(generated):
            allowed = self.func_trie.allowed_tokens(generated)
            next_token = min(allowed)
            generated.append(next_token)

        print(self.llm.decode(generated))
        print(self.func_trie.get_name(generated))

    def run(self):
        pass
