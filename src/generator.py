from typing import Dict, List, Set, Tuple
from enum import Enum
from llm_sdk import Small_LLM_Model
from .file_handling import Function, CallResult
from .utils import PromptBuilder, to_type
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


def mask_logits(logits: List[float], allowed: Set[int]) -> List[float]:
    for token_id in range(len(logits)):
        if token_id not in allowed:
            logits[token_id] = float("-inf")
    return logits


class State(Enum):
    SELECT_FUNCTION = 1
    SELECT_PARAM = 2
    SELECT_VALUE = 3
    DONE = 4


class Generator():
    def __init__(self, functions: List[Function]) -> None:
        self.llm = Small_LLM_Model()
        self.functions = functions
        self.input_ids: List[int] = []
        self.max_tokens = 15
        self.state = State.SELECT_FUNCTION
        self.curr_func: Function
        self.remaining_params = []
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
        self.gen_configs: Dict[str, Tuple[Set[int], Tuple[str, ...]]] = {
            "string": (self.str_allowed, ('"',)),
            "number": (self.num_allowed, (",", "\n")),
            "integer": (self.int_allowed, (",", "\n"))
        }

    def _get_vocab(self) -> Dict[str, int]:
        path = self.llm.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return vocab

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

    def _add_to_trie(self, trie: PrefixTrie, to_encode: str) -> None:
        token_ids = self.llm.encode(to_encode)[0].tolist()
        trie.insert(token_ids, to_encode)

    def _inject(self, to_encode: str) -> None:
        token_ids: List[int] = self.llm.encode(to_encode)[0].tolist()
        self.input_ids.extend(token_ids)
        print(self.llm.decode(token_ids), end="")

    def _generate_from_trie(self, trie: PrefixTrie) -> List[int]:
        generated: List[int] = []

        while not trie.is_complete(generated):
            allowed = trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)
            print(self.llm.decode([next_token]), end="")

        return generated

    def _generate_from_mask(
        self,
        allowed: Set[int],
        stop_chars: Tuple[str, ...]
    ) -> List[int]:
        generated: List[int] = []

        while len(generated) < self.max_tokens:
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = mask_logits(logits, allowed)
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

        return generated

    def _gen_func_name(self) -> str:
        funcs_trie = PrefixTrie()

        for func in self.functions:
            self._add_to_trie(funcs_trie, func.name)

        func_name_ids = self._generate_from_trie(funcs_trie)

        return self.llm.decode(func_name_ids)

    def _gen_param_name(self) -> str:
        param_trie = PrefixTrie()

        next_param = self.remaining_params[0]
        self._add_to_trie(param_trie, next_param)

        param_name_ids = self._generate_from_trie(param_trie)

        self.remaining_params.remove(next_param)

        return self.llm.decode(param_name_ids)

    def _gen_param_value(self, p_type: str) -> str:
        if p_type == "boolean":
            bool_trie = PrefixTrie()
            bool_vals = ["true", "false"]

            for val in bool_vals:
                self._add_to_trie(bool_trie, val)

            param_val_ids = self._generate_from_trie(bool_trie)

            return self.llm.decode(param_val_ids)
        else:
            allowed, stop_chars = self.gen_configs[p_type]

            param_val_ids = self._generate_from_mask(allowed, stop_chars)

            return self.llm.decode(param_val_ids)

    def run(self, user_prompt: str) -> CallResult:
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
                print(f"\nPrompt: {user_prompt}")
                self._inject('{\n  "name": "')

                func_name = self._gen_func_name()

                if func_name == "fn_none_found":
                    self._inject('",\n  "parameters": {}\n}')
                    return CallResult(
                        prompt=user_prompt,
                        name=func_name,
                        parameters=params
                    )

                self._inject('",\n  "parameters": {\n')

                for func in self.functions:
                    if func_name == func.name:
                        self.curr_func = func
                        self.remaining_params = list(func.parameters.keys())

                self.state = State.SELECT_PARAM

            if self.state == State.SELECT_PARAM:
                self._inject('    "')
                p_name = self._gen_param_name()

                p_type = self.curr_func.parameters[p_name].type
                if p_type in {"number", "integer"}:
                    self._inject('": ')
                else:
                    self._inject('": "')

                self.state = State.SELECT_VALUE

            if self.state == State.SELECT_VALUE:
                p_value = self._gen_param_value(p_type)

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
