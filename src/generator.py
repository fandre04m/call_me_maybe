from typing import Dict, List, Set, Tuple, Union
from llm_sdk import Small_LLM_Model
from .file_handling import Function, CallResult
from .utils import PromptBuilder, GenerationLogger, to_type
import json


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[int, TrieNode] = {}
        self.name: str | None = None


class PrefixTrie:
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


class VocabConstraints:
    def __init__(self, llm: Small_LLM_Model) -> None:
        self.vocab_file: Dict[str, int] = self._get_vocab(llm)
        self.str_allowed: Set[int] = self._make_mask(
            llm, to_forbid="`#$^"
        )
        self.num_allowed: Set[int] = self._make_mask(
            llm, to_allow=",\n0123456789-."
        )
        self.int_allowed: Set[int] = self._make_mask(
            llm, to_allow=",\n0123456789-"
        )
        self.gen_configs: Dict[str, Tuple[Set[int], Tuple[str, ...]]] = {
            "string": (self.str_allowed, ('"',)),
            "number": (self.num_allowed, (",", "\n")),
            "integer": (self.int_allowed, (",", "\n"))
        }
        pass

    def _get_vocab(self, llm: Small_LLM_Model) -> Dict[str, int]:
        path = llm.get_path_to_vocab_file()
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return vocab

    def _make_mask(
        self,
        llm: Small_LLM_Model,
        *,
        to_allow: str = "",
        to_forbid: str = ""
    ) -> Set[int]:
        allowed = []

        allow_set = set(to_allow)
        forbid_set = set(to_forbid)

        for t_id in self.vocab_file.values():
            t_text = llm.decode([t_id])

            if allow_set:
                if all(c in allow_set for c in t_text):
                    allowed.append(t_id)

            elif forbid_set:
                if all(c not in t_text for c in forbid_set):
                    allowed.append(t_id)

        return set(allowed)


class ConstrainedDecoder:
    def __init__(
        self,
        llm: Small_LLM_Model,
        logger: GenerationLogger,
        max_tokens: int = 15
    ):
        self.llm = llm
        self.logger = logger
        self.input_ids: List[int] = []
        self.max_tokens = max_tokens

    def add_to_trie(self, trie: PrefixTrie, to_encode: str) -> None:
        token_ids = self.llm.encode(to_encode)[0].tolist()
        trie.insert(token_ids, to_encode)

    def inject(self, to_encode: str) -> None:
        token_ids: List[int] = self.llm.encode(to_encode)[0].tolist()
        self.input_ids.extend(token_ids)
        self.logger.token(self.llm.decode(token_ids))

    def generate_from_trie(self, trie: PrefixTrie) -> List[int]:
        generated: List[int] = []

        while not trie.is_complete(generated):
            allowed = trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)
            self.logger.token(self.llm.decode([next_token]))

        return generated

    def generate_from_mask(
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
                prefix: str = tok_val[:stop_idx]
                if prefix:
                    prefix_ids: List[int] = self.llm.encode(prefix)[0].tolist()
                    self.input_ids.extend(prefix_ids)
                    generated.extend(prefix_ids)
                    self.logger.token(prefix)
                break

            self.input_ids.append(next_token)
            generated.append(next_token)
            self.logger.token(tok_val)

        return generated


class FunctionCalllGenerator:
    def __init__(
        self, functions: List[Function],
        llm: Small_LLM_Model
    ) -> None:
        self.functions = functions
        self.llm = llm
        self.logger = GenerationLogger()
        self.constraints = VocabConstraints(llm)
        self.decoder = ConstrainedDecoder(llm, self.logger)
        self.funcs_trie = self._gen_funcs_trie(self.functions)
        self.curr_func: Function
        self.remaining_params: List[str] = []

    def _gen_funcs_trie(self, functions: List[Function]) -> PrefixTrie:
        funcs_trie = PrefixTrie()

        for func in functions:
            self.decoder.add_to_trie(funcs_trie, func.name)

        return funcs_trie

    def _gen_func_name(self) -> str:
        func_name_ids = self.decoder.generate_from_trie(self.funcs_trie)

        return self.llm.decode(func_name_ids)

    def _gen_param_name(self) -> str:
        param_trie = PrefixTrie()

        next_param = self.remaining_params[0]
        self.decoder.add_to_trie(
            param_trie,
            next_param
        )

        param_name_ids = self.decoder.generate_from_trie(param_trie)

        self.remaining_params.remove(next_param)

        return self.llm.decode(param_name_ids)

    def _gen_param_value(self, p_type: str) -> str:
        if p_type == "boolean":
            bool_trie = PrefixTrie()
            bool_vals = ("true", "false")

            for val in bool_vals:
                self.decoder.add_to_trie(bool_trie, val)

            param_val_ids = self.decoder.generate_from_trie(bool_trie)

            return self.llm.decode(param_val_ids)
        else:
            allowed, stop_chars = self.constraints.gen_configs[p_type]

            param_val_ids = self.decoder.generate_from_mask(
                allowed,
                stop_chars
            )

            return self.llm.decode(param_val_ids)

    def run(self, user_prompt: str) -> CallResult:
        prompter = PromptBuilder()
        prompt = prompter.make_prompt(self.functions, user_prompt)
        self.decoder.input_ids = self.llm.encode(prompt)[0].tolist()

        self.logger.prompt(user_prompt)
        self.decoder.inject('{\n  "name": "')

        func_name = self._gen_func_name()
        params: Dict[str, Union[str, int, float, bool]] = {}

        if func_name == "fn_none_found":
            self.decoder.inject('",\n  "parameters": {}\n}')
            return CallResult(
                prompt=user_prompt,
                name=func_name,
                parameters=params
            )

        self.decoder.inject('",\n  "parameters": {\n')

        for func in self.functions:
            if func_name == func.name:
                self.curr_func = func
                self.remaining_params: List[str] = list(func.parameters.keys())

        while self.remaining_params:
            self.decoder.inject('    "')

            p_name = self._gen_param_name()
            p_type = self.curr_func.parameters[p_name].type

            is_num = p_type in {"number", "integer"}

            self.decoder.inject('": ' if is_num else '": "')

            p_value = self._gen_param_value(p_type)
            p_value = to_type(p_type, p_value)
            params[p_name] = p_value

            if self.remaining_params:
                self.decoder.inject(',\n' if is_num else '",\n')
            else:
                self.decoder.inject('\n  }\n}' if is_num else '"\n  }\n}')

        return CallResult(
            prompt=user_prompt,
            name=func_name,
            parameters=params,
        )
