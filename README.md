*This project has been created as part of the 42 curriculum by fandre-m.*

# Function Calling

## Description

This project implements a constrained decoding approach for function calling using a small language model.
Instead of generating unrestricted text, the decoder applies constraints during generation to 
ensure the output is a valid JSON object containing a function name and its corresponding parameters.

The project loads function definitions and user prompts from JSON files, generates a function call for
each prompt, and writes the results to an output JSON file.

---

## Features

- Validation of input files using Pydantic.
- Constrained function name generation using a prefix trie.
- Constrained parameter name generation also using prefix tries.
- Type-aware parameter value generation using vocabulary masks.
- Automatic conversion of generated values to Python types.
- JSON output containing generated function calls.

---

## Instructions

### Requirements

- Python 3.12+
- Required Python packages:
  - `pydantic`
  - `numpy`
  - Model dependencies for `Qwen3-0.6B`

### Installation

```bash
uv sync
```
or
```bash
make install
```

### Execution

Run with default input files:

```bash
uv run python3 -m src
```
or
```bash
make run
```

Or specify custom files:

```bash
uv run python3 -m src \
    --functions_definition path/to/functions.json \
    --input path/to/prompts.json \
    --output path/to/results.json
```

---

## Algorithm Explanation

The decoder generates the function call one field at a time while applying constraints to the
language model output.

Function names and parameter names are stored in prefix tries. During generation, only tokens that
continue a valid trie path remain available. Invalid tokens are masked by replacingtheir logits
with negative infinity before selecting the next token.
Generation stops once a value matching the one store in the last trie node is found.

Parameter values use vocabulary masks instead of tries. Allowed tokens are determined by the expected
parameter type, restricting generation to valid characters for strings, numbers, integers, and booleans.
Generation stops once a predefined terminating character is encountered.

This approach ensures that every generated token satisfies the current constraint while allowing the
language model to choose the most likely valid continuation.

---

## Design Decisions

- Prefix tries were chosen to easily guarantee valid function and parameter names.
- Vocabulary masks were used for parameter values since those cannot be fully constrained with the given data.
- Greedy decoding was selected for simplicity and deterministic output.
- Pydantic was used to validate input data before generation and correct output formatting.
- Generation logic was separated from validation and file handling to keep responsibilities independent.
- Eager caching of available function names and lazy caching of parameter names was used for performance optimizations.

---

## Performance Analysis

The constrained decoding approach guarantees that generated function and parameter names always match
the available definitions.

Trie lookups and vocabulary masking introduce little computational overhead compared to language model inference, making the solution efficient for the project requirements.

The implementation is reliable when provided with valid function definitions and prompts, as invalid JSON objects are prevented by injecting the correct JSON format tokens during each generation step.

---

## Challenges Faced

The main challenge was constraining language model output while preserving valid field generation.

Implementing prefix tries required maintaining valid token sequences throughout decoding. Parameter value
generation required a different strategy, leading to the use of vocabulary masks based on parameter types.
Getting reliable enough results to be able to stop generation correctly on parameter values was a decent
struggle, since we using such a small LLM.

Another big challenge I faced initially was a sub otpimal first research on LLM constrained decoding,
which led me in paths that could not obtain passable results. Those required big restructuring a few times
and made it so the final product was not exactly what I wanted to produce, even if it generates good results.

---

## Testing Strategy

The implementation was tested using multiple function definitions and prompts covering all supported
parameter types.

Additional tests included:

- Invalid function definitions.
- Invalid prompts.
- Empty input files.
- Malformed JSON files.
- Prompts that should produce no matching function.
- Generation of string, numeric, integer, and boolean parameters.

Generated outputs were manually verified to ensure correct parameter types and values.

---

## Example Usage

Example input:

```text
What is the sum of 4 and 8?
```

Example output:

```json
{
  "prompt": "What is the sum of 4 and 8?"
  "name": "fn_add_numbers",
  "parameters": {
    "a": 4,
    "b": 8
  }
}
```

---

## Resources

### References

- Qwen3-0.6B page: https://huggingface.co/Qwen/Qwen3-0.6B
- Efficient Guided Generation for Large Language Models: https://arxiv.org/pdf/2307.09702
- A Guide to Structured Generation Using Constrained Decoding: https://www.aidancooper.co.uk/constrained-decoding/

### AI Usage

AI was used in a multitude of ways:
Explaining concepts, since it is so much faster at finding resources online and explaining them.
Debugging, mainly producing edge cases and correcting hard to find bugs in the code.
Producing project related documentation, but mainly docstrings.

