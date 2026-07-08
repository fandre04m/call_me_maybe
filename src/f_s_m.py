from typing import Dict, List, Set
from enum import Enum
from llm_sdk import Small_LLM_Model
from .file_handling import Function, CallResult
from .utils import PromptBuilder, to_type
import time
import json


class TrieNode():
    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = {}
        self.name: str | None = None


class PrefixTrie():
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: List[int], name: str) -> None:
        current = self.root

        for t_id in token_ids:
            if t_id not in current.children:
                current.children[t_id] = TrieNode()
            current = current.children[t_id]

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
        self.llm = Small_LLM_Model()
        self.functions = functions
        self.max_tokens = 15
        self.elapsed_time: float = 0.0
        self.state = State.SELECT_FUNCTION
        self.curr_func: Function
        self.remaining_params = []
        self.input_ids: List[int] = []
        self.gen_ids: List[int] = []
        self.vocab_file: Dict[str, int] = self._get_vocab()
        self.str_allowed: Set[int] = self._make_mask(
            to_forbid="`#$^"
        )
        self.num_allowed: Set[int] = self._make_mask(
            to_allow=",\n0123456789-."
        )
        self.int_allowed: Set[int] = self._make_mask(
            to_allow=",\n0123456789-"
        )

    def _get_vocab(self) -> Dict[str, int]:
        path = self.llm.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return vocab

    def _add_to_trie(self, trie: PrefixTrie, to_encode: str) -> None:
        start = time.monotonic()
        token_ids = self.llm.encode(to_encode)[0].tolist()
        self.elapsed_time += time.monotonic() - start
        trie.insert(token_ids, to_encode)

    def _make_mask(
        self,
        *,
        to_allow: str = "",
        to_forbid: str = ""
    ) -> Set[int]:
        allowed = []

        allow_set = set(to_allow)
        forbid_set = set(to_forbid)

        for t_id in self.vocab_file.values():
            t_text = self.llm.decode([t_id])

            if allow_set:
                if all(c in allow_set for c in t_text):
                    allowed.append(t_id)

            elif forbid_set:
                if all(c not in t_text for c in forbid_set):
                    allowed.append(t_id)

        return set(allowed)

    def _mask_logits(
        self,
        logits: List[float],
        allowed: Set[int]
    ) -> List[float]:
        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = float("-inf")
        return logits

    def _inject(self, to_encode: str) -> None:
        start = time.monotonic()
        token_ids: List[int] = self.llm.encode(to_encode)[0].tolist()
        self.elapsed_time += time.monotonic() - start
        self.input_ids.extend(token_ids)
        self.gen_ids.extend(token_ids)
        print(self.llm.decode(token_ids), end="")

    def _gen_func_name(self) -> List[int]:
        prefix_trie = PrefixTrie()
        for func in self.functions:
            self._add_to_trie(prefix_trie, func.name)

        generated: List[int] = []
        start = time.monotonic()
        while not prefix_trie.is_complete(generated):
            allowed = prefix_trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = self._mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)
            print(self.llm.decode([next_token]), end="")

        if prefix_trie.get_name(generated) == "fn_no_match":
            raise ValueError(
                "Could not find a valid function for the prompt."
            )

        self.elapsed_time += time.monotonic() - start
        return generated

    def _gen_param_name(self) -> List[int]:
        prefix_trie = PrefixTrie()

        next_param = self.remaining_params[0]
        self._add_to_trie(prefix_trie, next_param)

        generated: List[int] = []
        start = time.monotonic()
        while not prefix_trie.is_complete(generated):
            allowed = prefix_trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = self._mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)
            print(self.llm.decode([next_token]), end="")

        self.elapsed_time += time.monotonic() - start
        self.remaining_params.remove(next_param)
        return generated

    def _gen_param_value(self, p_type: str) -> List[int]:
        stop_chars = set()
        allowed = set()
        if p_type == "string":
            allowed = self.str_allowed
            stop_chars = ('"',)
        if p_type == "number":
            allowed = self.num_allowed
            stop_chars = ("\n", ",")
        if p_type == "integer":
            allowed = self.int_allowed
            stop_chars = ("\n", ",")

        generated: List[int] = []
        start = time.monotonic()
        while len(generated) < self.max_tokens:
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = self._mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            tok_val = self.llm.decode([next_token])

            stop_idx = min(
                (tok_val.index(c) for c in stop_chars if c in tok_val),
                default=-1,
            )
            if stop_idx != -1:
                prefix = tok_val[:stop_idx]
                if prefix:
                    prefix_ids = self.llm.encode(prefix)[0].tolist()
                    self.input_ids.extend(prefix_ids)
                    generated.extend(prefix_ids)
                    print(prefix, end="")
                break

            self.input_ids.append(next_token)
            generated.append(next_token)
            print(tok_val, end="")

        self.elapsed_time += time.monotonic() - start
        return generated

    def run(self, user_prompt: str) -> CallResult:
        self.gen_ids = []

        prompter = PromptBuilder()
        prompt = prompter.make_prompt(self.functions, user_prompt)
        self.input_ids = self.llm.encode(prompt)[0].tolist()

        self.state = State.SELECT_FUNCTION
        func_name = ""
        p_name = ""
        p_value = ""
        p_type = ""
        params = {}
        while self.state != State.DONE:
            if self.state == State.SELECT_FUNCTION:
                print(f"Prompt: {user_prompt}")
                self._inject('{\n  "name": "')

                func_name_ids = self._gen_func_name()
                func_name = self.llm.decode(func_name_ids)
                self.gen_ids.extend(func_name_ids)

                self._inject('",\n  "parameters": {\n')

                for func in self.functions:
                    if func_name == func.name:
                        self.curr_func = func
                        self.remaining_params = list(func.parameters.keys())

                self.state = State.SELECT_PARAM

            if self.state == State.SELECT_PARAM:
                self._inject('    "')
                p_name_ids = self._gen_param_name()
                p_name = self.llm.decode(p_name_ids)
                self.gen_ids.extend(p_name_ids)

                p_type = self.curr_func.parameters[p_name].type
                if p_type in {"number", "integer"}:
                    self._inject('": ')
                else:
                    self._inject('": "')

                self.state = State.SELECT_VALUE

            if self.state == State.SELECT_VALUE:
                p_value_ids = self._gen_param_value(p_type)
                p_value = self.llm.decode(p_value_ids)
                self.gen_ids.extend(p_value_ids)

                p_value = to_type(p_type, p_value)
                params[p_name] = p_value

            if not self.remaining_params:
                if p_type in {"number", "integer"}:
                    self._inject('\n  }\n}')
                else:
                    self._inject('"\n  }\n}')
                self.state = State.DONE
            else:
                if p_type in {"number", "integer"}:
                    self._inject(',\n')
                else:
                    self._inject('",\n')
                self.state = State.SELECT_PARAM

        if self.state == State.DONE:
            return CallResult(
                prompt=user_prompt,
                name=func_name,
                parameters=params,
            )
        else:
            raise ValueError("Could not return a valid JSON object.")
