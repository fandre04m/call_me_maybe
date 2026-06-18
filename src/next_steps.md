
1. **Write the three extraction functions** (`extract_numbers`, `extract_ints`, `extract_strings`) plus a `STOP_WORDS` set, adapted from the peer's code, operating on the raw user prompt.

2. **Write `extract_params_options`**: dispatch by `param.type`, then filter out candidates that match the function name or any parameter name (so structural words don't get offered as values).

3. **Decide your per-parameter strategy split**: classify each param as either "extractable" (value should literally appear in the prompt — `source_string`, `a`, `b`, `name`, `replacement`) or "synthetic" (value must be inferred/constructed — `regex`). You can do this with a simple heuristic for now (type-based, or a manual flag on `Function`/`Param`), refine later.

4. **For extractable params**: build a `PrefixTrie` from the candidate list returned by `extract_params_options`, and reuse your existing constrained-decoding loop (same pattern as `gen_func_name`/`gen_param_name`) to have the model pick one trie-constrained value instead of generating freely.

5. **For synthetic params** (regex specifically): keep your current free-generation + sentinel-stop approach as a fallback. Optionally add a small curated keyword→pattern lookup (e.g. "vowels"→`[aeiou]`, "numbers"→`\d+`, "whitespace"→`\s+`) checked against the user prompt before falling back to generation, since pattern coverage for common asks is small and high-value.

6. **Handle empty candidate lists gracefully**: if extraction finds nothing for an extractable param, decide whether to fall back to free generation or raise/flag for clarification.

7. **Re-verify argument ordering**: once values are filled via this new path, make sure you're still reassembling `param_name → value` pairs in the function's declared order before treating the call as complete (the earlier a/b swap bug), since generation order is independent of declaration order regardless of which value strategy you use.

8. **Test on your existing prompt set**, especially the three regex/string-heavy cases, and check whether extraction alone fixes `source_string`/`replacement`, leaving `regex` as your only remaining accuracy gap to tackle separately.

Go through these and ask as you hit specifics.
