#!/usr/bin/env python3
from pathlib import Path
import argparse
from typing import List
from llm_sdk import Small_LLM_Model
from src import FileLoader, Generator, CallResult
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
    try:
        file_loader.load_func_definitions(Path(args.functions_definition))
        file_loader.load_prompts(Path(args.input))
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON - {e}")
    except (PermissionError, OSError) as e:
        print(f"Error: file system - {e}")

    llm = Small_LLM_Model()
    fsm = Generator(file_loader.func_definitions, llm)
    results: List[CallResult] = []
    # res: CallResult = fsm.run("")
    for prompt in file_loader.prompts:
        try:
            res: CallResult = fsm.run(prompt.prompt)
            results.append(res)
            print("\nSuccess!")
            # print(f"Elapsed time: {fsm.elapsed_time:.2f}")
        except ValueError as e:
            print(f"\nError: {e}")
            continue
    file_loader.write_call_result(results, Path(args.output))


main()
