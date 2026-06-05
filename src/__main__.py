#!/usr/bin/env python3
from pathlib import Path
import argparse
from src import FileLoader, PromptBuilder, LLMHandler, PrefixTrie
from pydantic import ValidationError
import json


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    file_loader = FileLoader()
    # prompt_maker = PromptBuilder()
    model = LLMHandler()
    func_trie = PrefixTrie()
    try:
        file_loader.load_func_definitions(Path(args.functions_definition))
        file_loader.load_prompts(Path(args.input))
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON - {e}")
    except ValidationError as e:
        print(f"Error: data not compatible - {e}")
    except (PermissionError, OSError) as e:
        print(f"Error: file system - {e}")

    # for p in file_handler.prompts:
    # built_prompt = prompt_maker.build(
    #     file_loader.func_definitions,
    #     file_loader.prompts[0].prompt
    # )
    for func in file_loader.func_definitions:
        token_ids = model.get_func_name_tokens(func.name)
        func_trie.insert(token_ids, func.name)

    # model.param_name_tokens()
    # print(f"\nPrompt: {file_handler.prompts[0].prompt}\n")
    # model.run_prompt(built_prompt)
    # print(f"Elapsed time: {model.elapsed:.2f} seconds")
    # file_handler.write_call_result(mock_results, Path(args.output))


main()
